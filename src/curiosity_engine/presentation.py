from __future__ import annotations

from typing import Any


def format_learning_thread(result: dict[str, Any]) -> str:
    output = result.get("output") or {}
    extension = output.get("physical_extension") or {}
    reasoning = output.get("_reasoning") or {}
    parts = [
        f"*Start here*\n{output.get('hook', 'Follow the question together.')}",
        f"*Show*\n{output.get('show', 'Notice what you can observe together.')}",
        f"*Ask*\n{output.get('ask', 'What do you notice?')}",
        f"*Tiny explanation*\n{output.get('nugget', 'Keep the explanation small and follow the next question.')}",
    ]
    if extension:
        materials = extension.get("materials") or []
        instructions = extension.get("instructions") or []
        physical = [f"*Try it: {extension.get('title', 'A quick investigation')}*"]
        if materials:
            physical.append("Materials: " + ", ".join(str(item) for item in materials))
        physical.extend(f"{index}. {step}" for index, step in enumerate(instructions, start=1))
        parts.append("\n".join(physical))
    public_sources = [
        str(item)
        for item in output.get("resource_refs") or []
        if str(item).startswith(("https://", "http://"))
    ]
    if public_sources:
        parts.append("*Sources*\n" + "\n".join(f"• <{url}>" for url in public_sources[:3]))
    if reasoning.get("backend") == "deterministic":
        parts.insert(0, "_Offline demo response — connect a reasoning provider for tailored answers._")
    if result.get("visual_job_id"):
        parts.append("_A visual is being prepared and will follow this answer._")
    return "\n\n".join(parts)[:4_000]


def response_did_not_pass() -> str:
    return (
        "I could not produce a reliable answer, so I stopped instead of showing a flawed draft. The diagnostic is "
        "saved privately on the computer running Curiosity Engine; you do not need to keep retyping the question. "
        "Run `curiosity doctor` on that computer to see the redacted answer-quality status."
    )
