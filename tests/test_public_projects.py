from __future__ import annotations

from datetime import date

import pytest

from curiosity_engine.public_projects import (
    audit_public_projects,
    load_public_project_registry,
    public_project_catalog,
    registry_status,
)


def test_curated_registry_is_current_unique_and_non_executable_by_default():
    registry = load_public_project_registry()
    assert len(registry.projects) == 14
    assert len({project.id for project in registry.projects}) == len(registry.projects)
    assert registry.default_policy == "reference_only"
    assert registry_status(today=date(2026, 8, 24))["status"] == "current"
    assert registry_status(today=date(2026, 11, 23))["status"] == "review_due"
    for project in registry.projects:
        if project.status == "approved_reference":
            assert project.use_mode in {"reference_only", "published_outputs_only"}
        if project.status in {"evaluation_candidate", "watch_only"}:
            assert project.use_mode in {"evaluation_only", "never_execute"}


def test_catalog_filters_researched_ocr_candidates_without_installing_anything():
    catalog = public_project_catalog(category="visual_ocr")
    assert {project["id"] for project in catalog["projects"]} == {
        "ocrmypdf",
        "paddleocr",
        "tesseract",
    }
    assert all(project["use_mode"] != "integrated_pinned_release" for project in catalog["projects"])


def test_live_audit_is_explicit_and_only_reads_public_metadata(monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="--live"):
        audit_public_projects(live=False)

    registry = load_public_project_registry()
    licenses = {
        project.metadata_source.identifier: project.license_spdx for project in registry.projects
    }
    requested: list[str] = []

    def fake_request(url: str, *, timeout: float):
        assert timeout == 2
        requested.append(url)
        if url.startswith("https://api.github.com/repos/"):
            identifier = url.removeprefix("https://api.github.com/repos/")
            return {
                "full_name": identifier,
                "archived": False,
                "disabled": False,
                "pushed_at": "2026-08-01T00:00:00Z",
                "license": {"spdx_id": licenses[identifier]},
            }
        assert url == "https://pypi.org/pypi/reportlab/json"
        return {"info": {"version": "5.0.0", "requires_python": ">=3.9", "yanked": False}}

    monkeypatch.setattr("curiosity_engine.public_projects._request_json", fake_request)
    report = audit_public_projects(live=True, timeout=2)
    assert report["status"] == "pass"
    assert report["family_data_sent"] is False
    assert report["upstream_code_downloaded"] is False
    assert report["upstream_code_executed"] is False
    assert len(requested) == len(registry.projects)
    assert all(
        url.startswith(("https://api.github.com/repos/", "https://pypi.org/pypi/"))
        for url in requested
    )
