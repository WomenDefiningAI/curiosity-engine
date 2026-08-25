from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from curiosity_engine.backups import (
    BackupError,
    backup_status,
    create_snapshot,
    default_backup_root,
    restore_snapshot,
    verify_snapshot,
)
from curiosity_engine.cli import main
from curiosity_engine.db import connect, init_db
from curiosity_engine.graph import add_child


def _family_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    private = repo / "private"
    output = private / "output"
    database = private / "data" / "curiosity.db"
    backup_root = tmp_path / "family-backups"
    (private / "resources" / "unit-one").mkdir(parents=True)
    output.mkdir(parents=True)
    (private / "setup").mkdir(parents=True)
    (private / "resources" / "unit-one" / "lesson.txt").write_text("licensed family material")
    (private / "resources" / "unit-one" / "lesson.pdf").write_bytes(b"synthetic-pdf")
    (output / "response.png").write_bytes(b"generated-family-image")
    (private / "setup" / "brain.json").write_text('{"provider":"example"}')
    (private / "setup" / "status.json").write_text('{"ready":true}')
    (private / "setup" / "model.env").write_text("OPENAI_API_KEY=sk-test-family-secret")
    (private / "setup" / "slack.env").write_text("SLACK_BOT_TOKEN=xoxb-test-family-secret")
    init_db(database)
    add_child(database, "kid-a", "Synthetic Child", 2019, "1")
    with connect(database) as connection:
        connection.execute(
            """INSERT INTO artifacts(id,child_id,artifact_type,path,spec_json,created_at)
               VALUES('artifact-a','kid-a','worksheet',?,'{}','2026-01-01T00:00:00+00:00')""",
            (str(output / "response.png"),),
        )
        connection.execute(
            """INSERT INTO resource_collections(id,title,provider,license_scope)
               VALUES('collection-a','Synthetic Collection','example','family_private')"""
        )
        connection.execute(
            """INSERT INTO resource_units(id,collection_id,title,ordinal)
               VALUES('unit-a','collection-a','Synthetic Unit',1)"""
        )
        connection.execute(
            """INSERT INTO resource_documents(
                 id,unit_id,title,document_type,source_path,sha256,content_access,metadata_json,indexed_at
               ) VALUES('document-a','unit-a','Synthetic Document','lesson',?,'abc','selected_excerpts',?,
                        '2026-01-01T00:00:00+00:00')""",
            (
                str(private / "resources" / "unit-one" / "lesson.pdf"),
                json.dumps({"text_extract_path": str(private / "resources" / "unit-one" / "lesson.txt")}),
            ),
        )
    return repo, private, database, backup_root


def _create(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    repo, private, database, backup_root = _family_fixture(tmp_path)
    result = create_snapshot(
        db_path=database,
        private_dir=private,
        output_dir=private / "output",
        backup_root=backup_root,
        repo_root=repo,
    )
    return repo, private, backup_root, result


def test_full_snapshot_is_external_owner_only_and_excludes_credentials(tmp_path: Path):
    repo, _private, backup_root, result = _create(tmp_path)
    snapshot = backup_root / str(result["snapshot"])

    assert not snapshot.is_relative_to(repo)
    assert (snapshot / "data" / "curiosity.db").is_file()
    assert (snapshot / "resources" / "unit-one" / "lesson.txt").is_file()
    assert (snapshot / "output" / "response.png").is_file()
    assert (snapshot / "setup" / "brain.json").is_file()
    assert (snapshot / "setup" / "status.json").is_file()
    assert not (snapshot / "setup" / "model.env").exists()
    assert not (snapshot / "setup" / "slack.env").exists()

    all_text = b"".join(item.read_bytes() for item in snapshot.rglob("*") if item.is_file())
    assert b"sk-test-family-secret" not in all_text
    assert b"xoxb-test-family-secret" not in all_text
    assert result["credentials_included"] is False
    assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700
    assert all(
        stat.S_IMODE(item.stat().st_mode) == (0o700 if item.is_dir() else 0o600)
        for item in [backup_root, snapshot, *snapshot.rglob("*")]
    )
    assert verify_snapshot(backup_root=backup_root, repo_root=repo)["status"] == "pass"


def test_status_is_summary_only_and_default_root_is_beside_repo(tmp_path: Path):
    repo, _private, backup_root, result = _create(tmp_path)
    status = backup_status(backup_root=backup_root, repo_root=repo)
    rendered = json.dumps(status)

    assert status["snapshots"] == 1
    assert status["latest"]["snapshot"] == result["snapshot"]
    assert "Synthetic Child" not in rendered
    assert "licensed family material" not in rendered
    assert default_backup_root(repo) == repo.parent / "repo-family-backups"


def test_verify_detects_tampering(tmp_path: Path):
    repo, _private, backup_root, result = _create(tmp_path)
    lesson = backup_root / str(result["snapshot"]) / "resources" / "unit-one" / "lesson.txt"
    lesson.write_text("changed after backup")
    lesson.chmod(0o600)

    verification = verify_snapshot(backup_root=backup_root, repo_root=repo)

    assert verification["status"] == "fail"
    assert any("mismatch" in error for error in verification["errors"])


def test_restore_verifies_and_never_overwrites(tmp_path: Path):
    repo, private, backup_root, result = _create(tmp_path)
    database = private / "data" / "curiosity.db"
    add_child(database, "kid-b", "Later Synthetic Child", 2018, "2")
    target = private / "restored-copy"

    restored = restore_snapshot(
        str(result["snapshot"]),
        target_private=target,
        backup_root=backup_root,
        repo_root=repo,
    )

    assert restored["active"] is False
    assert restored["credentials_restored"] is False
    with connect(target / "data" / "curiosity.db") as connection:
        children = [row[0] for row in connection.execute("SELECT id FROM children ORDER BY id")]
        artifact_path = connection.execute("SELECT path FROM artifacts WHERE id='artifact-a'").fetchone()[0]
        resource_row = connection.execute(
            "SELECT source_path,metadata_json FROM resource_documents WHERE id='document-a'"
        ).fetchone()
    assert children == ["kid-a"]
    assert artifact_path == str(target / "output" / "response.png")
    assert resource_row[0] == str(target / "resources" / "unit-one" / "lesson.pdf")
    assert json.loads(resource_row[1])["text_extract_path"] == str(
        target / "resources" / "unit-one" / "lesson.txt"
    )
    assert restored["database_paths_rebased"] == 2
    assert not (target / "setup" / "model.env").exists()
    with pytest.raises(BackupError, match="already exists"):
        restore_snapshot(
            str(result["snapshot"]),
            target_private=target,
            backup_root=backup_root,
            repo_root=repo,
        )


def test_restore_refuses_a_snapshot_that_no_longer_verifies(tmp_path: Path):
    repo, private, backup_root, result = _create(tmp_path)
    snapshot = backup_root / str(result["snapshot"])
    (snapshot / "output" / "response.png").write_bytes(b"tampered")
    (snapshot / "output" / "response.png").chmod(0o600)

    with pytest.raises(BackupError, match="verification failed"):
        restore_snapshot(
            str(result["snapshot"]),
            target_private=private / "should-not-exist",
            backup_root=backup_root,
            repo_root=repo,
        )
    assert not (private / "should-not-exist").exists()


def test_backup_refuses_repository_destination_and_source_symlinks(tmp_path: Path):
    repo, private, database, _backup_root = _family_fixture(tmp_path)
    with pytest.raises(BackupError, match="outside the repository"):
        create_snapshot(
            db_path=database,
            private_dir=private,
            output_dir=private / "output",
            backup_root=private / "backups",
            repo_root=repo,
        )
    link = private / "resources" / "linked.txt"
    link.symlink_to(private / "resources" / "unit-one" / "lesson.txt")
    with pytest.raises(BackupError, match="symbolic links"):
        create_snapshot(
            db_path=database,
            private_dir=private,
            output_dir=private / "output",
            backup_root=tmp_path / "external-backups",
            repo_root=repo,
        )


def test_backup_refuses_unsafe_destination_and_output_layouts(tmp_path: Path):
    repo, private, database, _backup_root = _family_fixture(tmp_path)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated.txt").write_text("do not modify")

    with pytest.raises(BackupError, match="initialized"):
        create_snapshot(
            db_path=database,
            private_dir=private,
            output_dir=private / "output",
            backup_root=occupied,
            repo_root=repo,
        )
    with pytest.raises(BackupError, match="setup credentials"):
        create_snapshot(
            db_path=database,
            private_dir=private,
            output_dir=private,
            backup_root=tmp_path / "external-backups-a",
            repo_root=repo,
        )
    (private / "output" / "accidental.env").write_text("API_KEY=secret")
    with pytest.raises(BackupError, match="credential-like"):
        create_snapshot(
            db_path=database,
            private_dir=private,
            output_dir=private / "output",
            backup_root=tmp_path / "external-backups-b",
            repo_root=repo,
        )

def test_backup_cli_create_status_and_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys):
    repo, private, database, backup_root = _family_fixture(tmp_path)
    monkeypatch.setenv("CURIOSITY_REPO_ROOT", str(repo))

    assert main(
        [
            "backup",
            "create",
            "--db",
            str(database),
            "--private-dir",
            str(private),
            "--output-dir",
            str(private / "output"),
            "--destination",
            str(backup_root),
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["status"] == "created"

    assert main(["backup", "status", "--destination", str(backup_root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "available"
    assert main(["backup", "verify", "--destination", str(backup_root)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
