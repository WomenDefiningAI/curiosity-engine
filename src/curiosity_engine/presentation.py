from __future__ import annotations

import re
from typing import Any


def _public_sources(output: dict[str, Any]) -> list[str]:
    sources = [
        str(item)
        for item in output.get("resource_refs") or []
        if str(item).startswith(("https://", "http://"))
    ]
    return list(dict.fromkeys(sources))[:3]


_GENERIC_LEAD_IN = re.compile(
    r"^(?:(?:sure|okay|ok|absolutely|of course)|"
    r"(?:(?:that['’]?s|that is|this is)\s+)?(?:a\s+)?(?:really\s+)?"
    r"(?:great|good|interesting|wonderful|smart) question)\s*[!,.\-—:]*\s*",
    re.IGNORECASE,
)
_GENERIC_RESPONSE_ONLY = re.compile(
    r"^(?:"
    r"(?:great|good|interesting) question|"
    r"let['’]?s look closely|"
    r"let['’]?s (?:look(?: closely| at (?:that|this|it))?|think about (?:that|this|it)|"
    r"explore (?:that|this|it)|find out|figure (?:that|this|it) out|learn about (?:that|this|it)) together|"
    r"follow the question together|"
    r"here is (?:a concise answer|an answer) to that question|"
    r"what do you think(?: about (?:that|this|it))?|"
    r"let['’]?s keep going"
    r")\s*[.!?\u2026]*$",
    re.IGNORECASE,
)

_GENERIC_TRANSITION_CLAUSE = re.compile(
    r"^(?:"
    r"(?:sure|okay|ok|absolutely|of course)|"
    r"(?:amazing|awesome|cool|exciting|fascinating|good|great|interesting|nice|wonderful)|"
    r"(?:(?:that['’]?s|that is|this is)\s+)?(?:a\s+)?(?:really\s+)?"
    r"(?:great|good|interesting|wonderful|smart) question|"
    r"(?:i(?:'d| would|'m| am)\s+)?(?:be\s+)?(?:happy|glad) to (?:help|assist)|"
    r"i(?:'m| am) excited to explore this with you|"
    r"thanks for asking|ready when you are|here we go|"
    r"(?:we(?:['’]?ll| will)|let['’]?s|let us) work (?:it|this|that) out together|"
    r"let['’]?s (?:dive|dig|jump|get) (?:right )?in|"
    r"let['’]?s (?:get started|unpack (?:that|this|it))|"
    r"let['’]?s (?:explore|look|think|find out|figure it out|learn)(?: about)?"
    r"(?: (?:that|this|it))?(?: together)?|"
    r"what do you think(?: about (?:that|this|it))?|"
    r"let['’]?s keep going"
    r")$",
    re.IGNORECASE,
)

_GENERIC_META_CLAUSE = re.compile(
    r"^(?:"
    r"(?:(?:this|that|it)|(?:[\w’'\-]+(?:\s+[\w’'\-]+){0,5}))\s+"
    r"(?:is|are|seem|seems|sound|sounds|feel|feels)\s+"
    r"(?!(?:not|never)\b)(?:[\w’'\-]+\s+){0,2}"
    r"(?:amazing|awesome|exciting|fascinating|fun|great|important|interesting|intriguing|wonderful)|"
    r"what\s+(?:an?\s+)?(?:(?:quite|really|so|truly|very)\s+)?"
    r"(?:amazing|awesome|exciting|fascinating|great|important|interesting|intriguing|wonderful)\s+"
    r"(?:idea|question|subject|thing|topic)|"
    r"(?:this|that|it)\s+"
    r"(?:is|are|seems|sounds|feels)\s+"
    r"(?:(?:a|an)\s+)?(?:(?:quite|really|so|truly|very)\s+)?"
    r"(?:amazing|awesome|exciting|fascinating|great|important|interesting|intriguing|thoughtful|wonderful)\s+"
    r"(?:idea|information|question|response|subject|thing|topic)|"
    r"here\s+(?:is|comes)\s+(?:(?:a|an|the)\s+)?(?:[\w’'\-]+\s+){0,4}"
    r"(?:answer|idea|response|thought|way)|"
    r"(?:we|i)(?:['’]?ll|\s+(?:will|can|should))?\s+"
    r"(?:consider|explore|investigate|review|think\s+about)\s+"
    r"(?:this|that|it|(?:(?:this|that|the)\s+)?(?:[\w’'\-]+\s+){0,3}"
    r"(?:idea|question|response|subject|thing|topic))(?:\s+(?:carefully|further|together))?|"
    r"(?:(?:this|that|the)\s+)?(?:idea|question|response|subject|thing|topic)\s+"
    r"(?:deserves|needs|requires|is\s+worth)\s+(?:[\w’'\-]+\s+){0,3}"
    r"(?:attention|consideration|exploration|thought)|"
    r"(?:this|that)\s+is\s+worth\s+"
    r"(?:considering|exploring|investigating|reviewing|thinking\s+about)\s+"
    r"(?:carefully|further|together)|"
    r"let(?:['’]?s|\s+us)\s+"
    r"(?:take\s+(?:(?:a|another)\s+)?(?:(?:really|very)\s+)?(?:close|closer|careful)\s+look|"
    r"work\s+(?:on|through)\s+(?:it|this|that)|"
    r"(?:look|think|consider|explore|investigate|learn|review|find\s+out|"
    r"figure\s+(?:it|this|that)\s+out)"
    r"(?:\s+(?:at|about))?\s*(?:it|this|that)?(?:\s+(?:carefully|further|together))?)"
    r")$",
    re.IGNORECASE,
)

def is_generic_response_only(text: str, question: str = "") -> bool:
    """Identify a transport-like transition mistakenly returned as the answer."""

    normalized = " ".join(text.split()).strip()
    without_lead_in = _GENERIC_LEAD_IN.sub("", normalized, count=1)
    if _GENERIC_RESPONSE_ONLY.fullmatch(normalized) or _GENERIC_RESPONSE_ONLY.fullmatch(without_lead_in):
        return True
    if _GENERIC_TRANSITION_CLAUSE.fullmatch(without_lead_in.strip(" \t\r\n,.:;!?—–-")):
        return True
    clauses = [
        clause.strip(" \t\r\n,.:;!?—–-")
        for clause in re.split(r"[.!?…;—–]+", normalized)
        if clause.strip(" \t\r\n,.:;!?—–-")
    ]
    if bool(clauses) and all(_GENERIC_TRANSITION_CLAUSE.fullmatch(clause) for clause in clauses):
        return True
    if bool(clauses) and all(_GENERIC_META_CLAUSE.fullmatch(clause) for clause in clauses):
        return True
    return not normalized


def _balanced_spoken_text(text: str) -> str:
    """Do not hand a parent a spoken sentence with a visibly broken quote."""

    cleaned = text.strip()
    if cleaned.count("“") > cleaned.count("”"):
        cleaned += "”"
    if cleaned.count('"') % 2:
        cleaned += '"'
    return cleaned


def _explicitly_concise(request: str) -> bool:
    return bool(
        re.search(
            r"\b(?:quick answer|keep it (?:very )?short|much shorter|brief answer|say (?:it|this) aloud|"
            r"answer (?:i|we) can say|what should i say|calm answer|only one sentence|"
            r"without starting over|just that part|only that part|"
            r"(?:one|two|three|1|2|3) sentences?)\b",
            request.casefold(),
        )
    )


def _requested_sentence_count(request: str) -> int | None:
    match = re.search(
        r"\b(?:keep (?:it )?to\s+|only\s+|just\s+)?(one|two|three|1|2|3)\s+sentences?\b",
        request.casefold(),
    )
    if not match:
        return None
    token = match.group(1)
    return int(token) if token.isdigit() else {"one": 1, "two": 2, "three": 3}[token]


def _first_sentences(text: str, count: int) -> str:
    sentences = [part.strip() for part in re.findall(r"[^.!?]+[.!?]", text) if part.strip()]
    return " ".join(sentences[:count]).strip() or text.strip()


def format_quick_answer(result: dict[str, Any], *, request: str = "") -> str:
    """Render a reviewed child answer without turning it into a miniature lesson."""

    output = result.get("output") or {}
    reasoning = output.get("_reasoning") or {}
    raw_answer = str(output.get("answer") or "").strip()
    if not raw_answer or is_generic_response_only(raw_answer):
        return response_did_not_pass()
    answer = _balanced_spoken_text(raw_answer)
    sentence_count = _requested_sentence_count(request)
    if sentence_count:
        answer = _balanced_spoken_text(_first_sentences(answer, sentence_count))
    parts = [answer]
    concise = _explicitly_concise(request)
    if output.get("notice") and not concise:
        parts.append(str(output["notice"]))
    question_back = str(output.get("question_back") or "").strip()
    rendered_so_far = "\n".join(parts).casefold()
    if question_back and not concise and question_back.casefold() not in rendered_so_far:
        parts.append(question_back)
    sources = _public_sources(output)
    if sources and not concise:
        parts.append("*Sources*\n" + "\n".join(f"• <{url}>" for url in sources))
    if reasoning.get("backend") == "deterministic":
        parts.insert(0, "_Offline demo response — connect a reasoning provider for tailored answers._")
    if result.get("visual_job_id") and not concise:
        parts.append("_A visual is being prepared and will follow this answer._")
    return "\n\n".join(part for part in parts if part).strip()[:4_000]


def format_learning_thread(result: dict[str, Any]) -> str:
    output = result.get("output") or {}
    extension = output.get("physical_extension") or {}
    reasoning = output.get("_reasoning") or {}
    response = str(output.get("parent_response") or "").strip()
    if response and extension.get("construction_pattern"):
        response = _first_sentences(response, 1)
    parts = [response] if response else []
    for heading, key in (
        ("Start here", "hook"),
        ("Show", "show"),
        ("Ask", "ask"),
        ("Tiny explanation", "nugget"),
    ):
        if output.get("parent_response") or (extension.get("construction_pattern") and key in {"show", "ask"}):
            continue
        value = str(output.get(key) or "").strip()
        if value:
            parts.append(f"*{heading}*\n{value}")
    if extension:
        materials = extension.get("materials") or []
        instructions = extension.get("instructions") or []
        prompts = extension.get("prompts") or []
        physical = [f"*Try it: {extension.get('title', 'A quick investigation')}*"]
        if materials:
            physical.append("Materials: " + ", ".join(str(item) for item in materials))
        physical.extend(f"{index}. {step}" for index, step in enumerate(instructions, start=1))
        if prompts:
            physical.append("*Ready-to-use prompts*")
            physical.extend(f"• {prompt}" for prompt in prompts)
        parts.append("\n".join(physical))
    public_sources = _public_sources(output)
    if public_sources:
        parts.append("*Sources*\n" + "\n".join(f"• <{url}>" for url in public_sources[:3]))
    if reasoning.get("backend") == "deterministic":
        parts.insert(0, "_Offline demo response — connect a reasoning provider for tailored answers._")
    if result.get("visual_job_id"):
        parts.append("_A visual is being prepared and will follow this answer._")
    return "\n\n".join(parts)[:4_000]


def format_reviewed_response(result: dict[str, Any]) -> str:
    """Choose the reviewed response shape from workflow provenance, never message keywords."""

    if result.get("status") is not None and result.get("status") != "completed":
        return response_did_not_pass()
    workflow = str(
        result.get("workflow")
        or (result.get("output") or {}).get("_reasoning", {}).get("policy", {}).get("workflow")
        or ""
    )
    output = result.get("output") or {}
    if workflow == "quick_answer" or "answer" in output:
        return format_quick_answer(result)
    return format_learning_thread(result)


def response_did_not_pass() -> str:
    return (
        "I couldn’t produce a reliable answer for this request. Say ‘try a simpler version’ and I’ll use your original "
        "request and the same supplies."
    )
