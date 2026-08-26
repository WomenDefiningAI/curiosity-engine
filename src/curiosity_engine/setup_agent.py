from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import family_home, family_workspace, private_root

SUPPORTED_AGENTS = ("codex", "claude")


def detect_agent(preference: str = "auto") -> str | None:
    if preference != "auto":
        return preference if shutil.which(preference) else None
    return next((name for name in SUPPORTED_AGENTS if shutil.which(name)), None)


def _write_owner_only(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(0o600)


def prepare_agent_setup(*, preference: str = "auto", workspace: str | Path | None = None) -> dict[str, Any]:
    """Create a private setup workspace an installed coding agent can safely operate in."""

    home = family_home()
    private = private_root()
    work = Path(workspace).expanduser().resolve() if workspace else family_workspace()
    for path in (home, private, work, private / "data", private / "output", private / "setup"):
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o700)

    agent = detect_agent(preference)
    agent_instructions = """# Curiosity Engine family setup workspace

You are the Family Operator for this Curiosity Engine installation.

Your job is to make the installed harness work for this family: inspect status, explain each decision in plain
language, run commands, test the Slack and model connections, and keep going until the owner has reviewed one real
family answer. Do not ask the parent to operate a terminal when you can safely do it. Never put family information,
purchased-resource content, or credentials in a public repository, commit, issue, or chat transcript.

Start by reading SETUP_HANDOFF.md and running `curiosity doctor`. Use `curiosity --help` to discover commands. Ask for
human input only for decisions, account creation, consent, credential entry into local owner-only files, and browser
steps you cannot complete safely. The parent must explicitly approve external writes and recurring schedules.

After a work session, act as the Open Source Steward too: identify generalizable, privacy-safe improvements. Propose
those separately from family-specific data. Never commit from this private workspace.
"""
    handoff = f"""# Setup handoff

This is private local configuration, not a source checkout.

- Family home: `{home}`
- Private data: `{private}`
- Setup workspace: `{work}`
- Detected coding agent: `{agent or 'none'}`

Begin with:

1. Run `curiosity doctor` and explain its next action.
2. Configure the household and children locally.
3. Help create and connect an adult-controlled Slack app; prove `connection` works without an LLM.
4. Help the family choose a reasoning + vision/OCR + image-generation model stack and run live probes.
5. Configure the family lens and explicitly decide whether licensed excerpts may be sent to the model provider.
6. Install the always-on local host services with `curiosity host install`.
7. Test one real Slack question, its visual, a free-form parent revision in-thread, and one generated printable.
8. Run `curiosity backup create`, verify it, and explain where the encrypted/off-device backup should live.

Never paste secrets into agent chat. `curiosity doctor` reports redacted status and the next checkpoint.
"""
    _write_owner_only(work / "AGENTS.md", agent_instructions)
    _write_owner_only(work / "SETUP_HANDOFF.md", handoff)
    state = {
        "version": 1,
        "family_home": str(home),
        "private_root": str(private),
        "workspace": str(work),
        "agent": agent,
        "credentials_present": False,
        "next": "launch_agent" if agent else "install_codex_or_claude_code",
    }
    _write_owner_only(private / "setup" / "agent-setup.json", json.dumps(state, indent=2) + "\n")
    return state


def launch_setup_agent(agent: str, workspace: str | Path) -> None:
    prompt = (
        "Read AGENTS.md and SETUP_HANDOFF.md completely. Act as my Family Operator. "
        "Start by running curiosity doctor and walk me through setup end to end."
    )
    work = str(Path(workspace).resolve())
    env = {**os.environ, "CURIOSITY_HOME": str(family_home())}
    if agent == "codex":
        command = [agent, "--cd", work, prompt]
    elif agent == "claude":
        command = [agent, "--permission-mode", "default", prompt]
    else:
        raise ValueError(f"unsupported coding agent: {agent}")
    subprocess.run(command, cwd=work, env=env, check=True)
