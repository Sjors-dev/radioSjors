"""Turn a chat message into an action.

Two outcomes matter: a *specific request* goes to the top-priority request
queue, a *vibe shift* updates the shared mood state the planner reads.  The LLM
does the classifying; a regex fallback keeps the bot useful when it is down.
"""

from __future__ import annotations

import logging
import re

from .llm import LLM, LLMUnavailable
from .prompts import INTENT_SYSTEM

log = logging.getLogger("intent")

_PLAY = re.compile(
    r"^\s*(?:!?play|put on|queue|speel|draai)\s+(.+?)\s*$", re.IGNORECASE)
_BY = re.compile(r"^(.*?)\s+(?:by|van|-|–)\s+(.+)$", re.IGNORECASE)

_VIBE_WORDS = (
    "make it", "something", "i want", "mood", "vibe", "darker", "lighter",
    "upbeat", "chill", "chiller", "calmer", "heavier", "softer", "faster",
    "slower", "sadder", "happier", "weirder", "mellow", "energetic", "harder",
    "less", "more", "meer", "minder", "rustiger", "harder",
)
_QUESTION_WORDS = (
    "what's playing", "whats playing", "what is playing", "now playing",
    "np", "queue", "what's next", "whats next", "coming up", "wat draait",
)


def classify(llm: LLM, text: str) -> dict:
    """Return {kind, artist, title, mood, reply, source}."""
    text = (text or "").strip()
    if not text:
        return {"kind": "chat", "reply": "", "source": "empty"}

    if llm.enabled:
        try:
            data = llm.complete_json(INTENT_SYSTEM, text, temperature=0.3,
                                     max_tokens=300)
            result = _normalize(data)
            if result:
                result["source"] = "llm"
                return result
        except LLMUnavailable as exc:
            log.info("intent LLM unavailable (%s), using rules", exc)
        except Exception as exc:
            log.warning("intent classification failed (%s), using rules", exc)

    result = _rules(text)
    result["source"] = "rules"
    return result


def _normalize(data: dict) -> dict | None:
    kind = str(data.get("kind") or "").strip().lower()
    if kind not in ("track", "vibe", "question", "chat"):
        return None
    return {
        "kind": kind,
        "artist": str(data.get("artist") or "").strip(),
        "title": str(data.get("title") or "").strip(),
        "mood": str(data.get("mood") or "").strip(),
        "reply": str(data.get("reply") or "").strip()[:300],
    }


def _rules(text: str) -> dict:
    lowered = text.lower()

    match = _PLAY.match(text)
    if match:
        target = match.group(1).strip().strip('"')
        by_match = _BY.match(target)
        if by_match:
            # "<title> by <artist>" is the common phrasing.
            title, artist = by_match.group(1).strip(), by_match.group(2).strip()
        else:
            title, artist = target, ""
        return {"kind": "track", "artist": artist, "title": title, "mood": "",
                "reply": f"Looking for {target}."}

    if any(word in lowered for word in _QUESTION_WORDS):
        return {"kind": "question", "artist": "", "title": "", "mood": "", "reply": ""}

    if any(word in lowered for word in _VIBE_WORDS):
        return {"kind": "vibe", "artist": "", "title": "", "mood": text,
                "reply": "Noted, shifting the mood."}

    return {"kind": "chat", "artist": "", "title": "", "mood": "", "reply": ""}
