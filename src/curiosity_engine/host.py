from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import family_home, family_workspace, private_root

SERVICE_NAMES = (
    "curiosity-engine-slack.service",
    "curiosity-engine-worker.service",
    "curiosity-engine-dashboard.service",
)


def _quote_systemd(value: str | Path) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _unit(command: list[str], *, description: str) -> str:
    home = family_home()
    work = home if (home / ".git").is_dir() else family_workspace()
    writable = private_root()
    rendered_command = " ".join(f'"{_quote_systemd(part)}"' for part in command)
    return f"""[Unit]
Description={description}
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
WorkingDirectory={_quote_systemd(work)}
Environment="CURIOSITY_HOME={_quote_systemd(home)}"
ExecStart={rendered_command}
Restart=on-failure
RestartSec=4
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths={_quote_systemd(writable)}
UMask=0077

[Install]
WantedBy=default.target
"""


def unit_definitions(executable: str | Path | None = None) -> dict[str, str]:
    command = str(Path(executable).resolve()) if executable else str(Path(sys.prefix) / "bin" / "curiosity")
    private = private_root()
    db = private / "data" / "curiosity.db"
    output = private / "output"
    return {
        "curiosity-engine-slack.service": _unit(
            [command, "slack", "run", "--db", str(db), "--output-dir", str(output)],
            description="Curiosity Engine Slack connector",
        ),
        "curiosity-engine-worker.service": _unit(
            [command, "worker", "--forever", "--db", str(db)],
            description="Curiosity Engine scheduled work runner",
        ),
        "curiosity-engine-dashboard.service": _unit(
            [
                command,
                "serve",
                "--db",
                str(db),
                "--output-dir",
                str(output),
                "--host",
                "127.0.0.1",
                "--port",
                "8766",
            ],
            description="Curiosity Engine private local review dashboard",
        ),
    }


def install_user_services(*, start: bool = True) -> dict[str, Any]:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        raise RuntimeError("always-on hosting currently requires Linux with systemd user services")
    unit_dir = Path.home() / ".config" / "systemd" / "user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_dir.chmod(0o700)
    private_root().mkdir(parents=True, exist_ok=True)
    private_root().chmod(0o700)
    if not (family_home() / ".git").is_dir():
        family_workspace().mkdir(parents=True, exist_ok=True)
        family_workspace().chmod(0o700)
    written: list[str] = []
    for name, content in unit_definitions().items():
        path = unit_dir / name
        path.write_text(content, encoding="utf-8")
        path.chmod(0o600)
        written.append(str(path))
    subprocess.run([systemctl, "--user", "daemon-reload"], check=True)
    if start:
        subprocess.run([systemctl, "--user", "enable", "--now", *SERVICE_NAMES], check=True)
    return {"status": "installed", "units": written, "started": start, "credentials_present": False}


def host_status() -> dict[str, Any]:
    systemctl = shutil.which("systemctl")
    if not systemctl:
        return {"supported": False, "reason": "systemd user services are unavailable"}
    services: dict[str, Any] = {}
    for name in SERVICE_NAMES:
        result = subprocess.run(
            [systemctl, "--user", "is-active", name],
            text=True,
            capture_output=True,
            check=False,
            env=os.environ,
        )
        services[name] = {"active": result.returncode == 0, "state": result.stdout.strip() or "unknown"}
    return {"supported": True, "services": services}
