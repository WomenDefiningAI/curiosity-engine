from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .config import configuration_root

ProjectStatus = Literal["integrated", "approved_reference", "evaluation_candidate", "watch_only"]
UseMode = Literal[
    "integrated_pinned_release",
    "reference_only",
    "published_outputs_only",
    "evaluation_only",
    "never_execute",
]
MetadataKind = Literal["github", "pypi"]


class MetadataSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: MetadataKind
    identifier: str = Field(min_length=1, max_length=160)


class PublicProject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")
    name: str = Field(min_length=1, max_length=120)
    publisher: str = Field(min_length=1, max_length=160)
    governance_notes: str = Field(min_length=1, max_length=600)
    categories: list[str] = Field(min_length=1, max_length=8)
    status: ProjectStatus
    use_mode: UseMode
    repository_url: HttpUrl
    documentation_url: HttpUrl
    freshness_url: HttpUrl
    metadata_source: MetadataSource
    license_spdx: str = Field(min_length=1, max_length=80)
    why_relevant: str = Field(min_length=1, max_length=600)
    age_fit: str = Field(min_length=1, max_length=500)
    safety_notes: list[str] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def status_matches_execution_policy(self) -> PublicProject:
        expected = {
            "integrated": {"integrated_pinned_release"},
            "approved_reference": {"reference_only", "published_outputs_only"},
            "evaluation_candidate": {"evaluation_only"},
            "watch_only": {"never_execute"},
        }
        if self.use_mode not in expected[self.status]:
            raise ValueError(f"{self.status} cannot use {self.use_mode}")
        if self.metadata_source.kind == "github" and "/" not in self.metadata_source.identifier:
            raise ValueError("GitHub metadata identifiers must be owner/repository")
        return self


class PublicProjectRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    reviewed_at: date
    recheck_after: date
    default_policy: Literal["reference_only"]
    policy_document: str
    projects: list[PublicProject] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_registry(self) -> PublicProjectRegistry:
        ids = [project.id for project in self.projects]
        if len(ids) != len(set(ids)):
            raise ValueError("public project IDs must be unique")
        if self.recheck_after <= self.reviewed_at:
            raise ValueError("recheck_after must be after reviewed_at")
        return self


def load_public_project_registry() -> PublicProjectRegistry:
    path = configuration_root() / "configs" / "public-projects.json"
    return PublicProjectRegistry.model_validate_json(path.read_text(encoding="utf-8"))


def public_project_catalog(
    *, category: str | None = None, status: ProjectStatus | None = None
) -> dict[str, Any]:
    registry = load_public_project_registry()
    projects = [
        project
        for project in registry.projects
        if (category is None or category in project.categories) and (status is None or project.status == status)
    ]
    return {
        "reviewed_at": registry.reviewed_at,
        "recheck_after": registry.recheck_after,
        "default_policy": registry.default_policy,
        "count": len(projects),
        "projects": [
            {
                "id": project.id,
                "name": project.name,
                "publisher": project.publisher,
                "governance_notes": project.governance_notes,
                "categories": project.categories,
                "status": project.status,
                "use_mode": project.use_mode,
                "repository_url": str(project.repository_url),
                "documentation_url": str(project.documentation_url),
                "freshness_url": str(project.freshness_url),
                "license_spdx": project.license_spdx,
                "why_relevant": project.why_relevant,
                "age_fit": project.age_fit,
                "safety_notes": project.safety_notes,
            }
            for project in projects
        ],
    }


def public_project(project_id: str) -> dict[str, Any]:
    catalog = public_project_catalog()
    match = next((item for item in catalog["projects"] if item["id"] == project_id), None)
    if match is None:
        raise KeyError(project_id)
    return {**match, "reviewed_at": catalog["reviewed_at"], "recheck_after": catalog["recheck_after"]}


def registry_status(*, today: date | None = None) -> dict[str, Any]:
    registry = load_public_project_registry()
    now = today or date.today()
    counts = {
        status: sum(project.status == status for project in registry.projects)
        for status in ("integrated", "approved_reference", "evaluation_candidate", "watch_only")
    }
    return {
        "status": "review_due" if now > registry.recheck_after else "current",
        "reviewed_at": registry.reviewed_at,
        "recheck_after": registry.recheck_after,
        "days_until_review": (registry.recheck_after - now).days,
        "default_policy": registry.default_policy,
        "counts": counts,
        "policy_document": registry.policy_document,
        "network_used": False,
    }


def _request_json(url: str, *, timeout: float) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "curiosity-engine-public-project-audit/1"}
    if url.startswith("https://api.github.com/") and os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['GITHUB_TOKEN']}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS APIs only
        return json.loads(response.read().decode("utf-8"))


def _github_check(project: PublicProject, *, timeout: float) -> dict[str, Any]:
    identifier = project.metadata_source.identifier
    data = _request_json(f"https://api.github.com/repos/{identifier}", timeout=timeout)
    concerns: list[str] = []
    if data.get("archived"):
        concerns.append("repository is archived")
    if data.get("disabled"):
        concerns.append("repository is disabled")
    pushed_at = data.get("pushed_at")
    if pushed_at:
        pushed = datetime.fromisoformat(str(pushed_at).replace("Z", "+00:00"))
        if datetime.now(UTC) - pushed > timedelta(days=365):
            concerns.append("no repository push observed in more than 365 days")
    observed_license = str((data.get("license") or {}).get("spdx_id") or "NOASSERTION")
    license_aliases = {
        "AGPL-3.0": "AGPL-3.0-only",
        "GPL-3.0": "GPL-3.0-only",
        "LGPL-3.0": "LGPL-3.0-only",
    }
    normalized_license = license_aliases.get(observed_license, observed_license)
    if normalized_license not in {project.license_spdx, "NOASSERTION"}:
        concerns.append(f"GitHub license changed or differs: {observed_license}")
    return {
        "id": project.id,
        "metadata_source": "github",
        "canonical_name": data.get("full_name"),
        "latest_push": pushed_at,
        "observed_license": observed_license,
        "archived": bool(data.get("archived")),
        "concerns": concerns,
        "status": "review_required" if concerns else "pass",
    }


def _pypi_check(project: PublicProject, *, timeout: float) -> dict[str, Any]:
    identifier = project.metadata_source.identifier
    data = _request_json(f"https://pypi.org/pypi/{identifier}/json", timeout=timeout)
    info = data.get("info") or {}
    concerns: list[str] = []
    if info.get("yanked"):
        concerns.append("latest package release is yanked")
    return {
        "id": project.id,
        "metadata_source": "pypi",
        "latest_release": info.get("version"),
        "requires_python": info.get("requires_python"),
        "concerns": concerns,
        "status": "review_required" if concerns else "pass",
    }


def audit_public_projects(*, live: bool, timeout: float = 10) -> dict[str, Any]:
    """Refresh public metadata without cloning, installing, importing, or executing upstream code."""

    if not live:
        raise ValueError("live audit contacts public repository APIs; pass --live explicitly")
    registry = load_public_project_registry()
    results: list[dict[str, Any]] = []
    for project in registry.projects:
        try:
            if project.metadata_source.kind == "github":
                result = _github_check(project, timeout=timeout)
            else:
                result = _pypi_check(project, timeout=timeout)
        except (OSError, urllib.error.HTTPError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
            result = {
                "id": project.id,
                "metadata_source": project.metadata_source.kind,
                "status": "error",
                "concerns": [f"metadata check failed: {type(exc).__name__}"],
            }
        results.append(result)
    return {
        "status": "review_required"
        if any(item["status"] != "pass" for item in results)
        else "pass",
        "checked_at": datetime.now(UTC).isoformat(),
        "network_used": True,
        "family_data_sent": False,
        "upstream_code_downloaded": False,
        "upstream_code_executed": False,
        "note": "Automated metadata is only a staleness alarm; it never replaces human license, security, and pedagogy review.",
        "results": results,
    }
