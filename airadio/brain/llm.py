"""Free-tier LLM client with failover between Gemini and Groq.

Both providers have genuinely free tiers with no card.  Hourly batching keeps
usage to a handful of calls per hour, far inside either limit, but a 429 or an
outage on one provider still falls straight through to the other.

Never send anything sensitive through here -- free tiers may train on prompts.
Music picks and DJ patter are fine.
"""

from __future__ import annotations

import json
import logging
import re
import time

import requests

from ..config import Config

log = logging.getLogger("llm")

GEMINI_URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
              "{model}:generateContent")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# How long a provider sits out after a rate limit before we try it again.
COOLDOWN_SECONDS = 300.0

# Completion budgets per job, overridable under llm.max_tokens in config.yaml.
# These are deliberately generous: reasoning models (Gemini 3.x, gpt-oss) spend
# part of the budget thinking before a single character of answer comes out, so
# a budget that looks ample for the visible reply returns an empty string.
DEFAULT_BUDGETS = {
    "plan": 8000,      # a full hour of running order plus every patter line
    "tag": 6000,       # mood/energy labels for a batch of tracks
    "intent": 2048,    # one short chat classification
    "probe": 2048,     # the doctor command's liveness check
}


class LLMUnavailable(RuntimeError):
    """Raised when every configured provider failed. Callers must degrade."""


class LLM:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.providers = [p.lower() for p in cfg.get("llm.providers", ["gemini", "groq"])]
        self.timeout = int(cfg.get("llm.timeout_seconds", 60))
        self.max_retries = int(cfg.get("llm.max_retries", 2))
        self.gemini_model = str(cfg.get("llm.gemini_model", "gemini-2.5-flash"))
        self.groq_model = str(cfg.get("llm.groq_model", "llama-3.3-70b-versatile"))
        self._cooldown: dict[str, float] = {}
        self._session = requests.Session()

        self._budgets = dict(DEFAULT_BUDGETS)
        for name, value in (cfg.get("llm.max_tokens") or {}).items():
            try:
                self._budgets[str(name)] = int(value)
            except (TypeError, ValueError):
                log.warning("ignoring non-numeric llm.max_tokens.%s: %r", name, value)

        # Off by default -- a full plan prompt runs to a few KB, every hour.
        # Turn on to see exactly what was asked for and what came back, e.g.
        # to compare a written conversation against the prompt that produced
        # it: journalctl -u ai-radio-brain | grep -A50 "LLM request (plan)"
        self.log_prompts = bool(cfg.get("llm.log_prompts", False))

    def budget(self, job: str) -> int:
        """Completion token budget for a named job."""
        return int(self._budgets.get(job, 4096))

    # -- keys ---------------------------------------------------------------

    def _key(self, provider: str) -> str:
        if provider == "gemini":
            return Config.env("GEMINI_API_KEY")
        if provider == "groq":
            return Config.env("GROQ_API_KEY")
        return ""

    @property
    def available_providers(self) -> list[str]:
        return [p for p in self.providers if self._key(p)]

    @property
    def enabled(self) -> bool:
        return bool(self.available_providers)

    # -- public API ---------------------------------------------------------

    def complete(self, system: str, user: str, json_mode: bool = False,
                 temperature: float = 0.8, max_tokens: int = 4096,
                 label: str = "") -> str:
        """Ask the first working provider. Raises LLMUnavailable if all fail."""
        now = time.time()
        errors: list[str] = []
        tag = f" ({label})" if label else ""

        if self.log_prompts:
            log.info("LLM request%s:\n--- system ---\n%s\n--- user ---\n%s",
                     tag, system, user)

        for provider in self.providers:
            key = self._key(provider)
            if not key:
                errors.append(f"{provider}: no API key")
                continue
            if self._cooldown.get(provider, 0) > now:
                wait = self._cooldown[provider] - now
                errors.append(f"{provider}: cooling down {wait:.0f}s")
                continue

            for attempt in range(self.max_retries + 1):
                try:
                    if provider == "gemini":
                        text = self._call_gemini(key, system, user, json_mode,
                                                 temperature, max_tokens)
                    elif provider == "groq":
                        text = self._call_groq(key, system, user, json_mode,
                                               temperature, max_tokens)
                    else:
                        errors.append(f"{provider}: unknown provider")
                        break

                    if text:
                        log.debug("%s answered (%d chars)", provider, len(text))
                        if self.log_prompts:
                            log.info("LLM response%s (%s, %d chars):\n%s",
                                    tag, provider, len(text), text)
                        return text
                    errors.append(f"{provider}: empty response")
                    break

                except _RateLimited as exc:
                    self._cooldown[provider] = time.time() + COOLDOWN_SECONDS
                    errors.append(f"{provider}: rate limited ({exc})")
                    break
                except _Retryable as exc:
                    if attempt >= self.max_retries:
                        errors.append(f"{provider}: {exc}")
                        break
                    backoff = 2 ** attempt
                    log.warning("%s transient error (%s), retrying in %ds",
                                provider, exc, backoff)
                    time.sleep(backoff)
                except Exception as exc:
                    errors.append(f"{provider}: {exc}")
                    break

        raise LLMUnavailable("; ".join(errors) or "no providers configured")

    def complete_json(self, system: str, user: str, temperature: float = 0.8,
                      max_tokens: int = 4096, label: str = "") -> dict:
        """Same as complete(), but insists on a JSON object coming back."""
        raw = self.complete(system, user, json_mode=True, temperature=temperature,
                            max_tokens=max_tokens, label=label)
        parsed = extract_json(raw)
        if parsed is None:
            raise LLMUnavailable(f"response was not JSON: {raw[:200]!r}")
        return parsed

    # -- providers ----------------------------------------------------------

    def _call_gemini(self, key: str, system: str, user: str, json_mode: bool,
                     temperature: float, max_tokens: int) -> str:
        url = GEMINI_URL.format(model=self.gemini_model)
        generation: dict = {"temperature": temperature,
                            "maxOutputTokens": max_tokens}
        if json_mode:
            generation["responseMimeType"] = "application/json"

        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": generation,
        }
        response = self._session.post(
            url, json=payload, timeout=self.timeout,
            headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        )
        self._raise_for_status("gemini", response)

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            feedback = data.get("promptFeedback") or {}
            raise _Retryable(f"no candidates ({feedback})")
        parts = ((candidates[0].get("content") or {}).get("parts") or [])
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            # Reasoning models spend the budget on thinking first, so an
            # empty reply usually means max_tokens was too small, not that
            # the model had nothing to say.
            reason = candidates[0].get("finishReason", "unknown")
            raise _Retryable(
                f"empty reply (finishReason={reason}); if this is MAX_TOKENS "
                f"the token budget of {max_tokens} is too small for a "
                f"reasoning model")
        return text

    def _call_groq(self, key: str, system: str, user: str, json_mode: bool,
                   temperature: float, max_tokens: int) -> str:
        payload: dict = {
            "model": self.groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        response = self._session.post(
            GROQ_URL, json=payload, timeout=self.timeout,
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"},
        )
        self._raise_for_status("groq", response)

        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise _Retryable("no choices returned")
        text = (choices[0].get("message") or {}).get("content", "").strip()
        if not text:
            reason = choices[0].get("finish_reason", "unknown")
            raise _Retryable(
                f"empty reply (finish_reason={reason}); if this is 'length' "
                f"the token budget of {max_tokens} is too small for a "
                f"reasoning model")
        return text

    @staticmethod
    def _raise_for_status(provider: str, response: requests.Response) -> None:
        if response.status_code == 200:
            return
        body = response.text[:300]
        if response.status_code == 429:
            raise _RateLimited(body)
        if response.status_code in (408, 500, 502, 503, 504):
            raise _Retryable(f"HTTP {response.status_code}: {body}")
        raise RuntimeError(f"HTTP {response.status_code}: {body}")


class _Retryable(RuntimeError):
    pass


class _RateLimited(RuntimeError):
    pass


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    """Pull a JSON object out of an LLM response, fenced or not."""
    if not text:
        return None
    text = text.strip()

    for candidate in ([m.group(1) for m in _FENCE.finditer(text)] + [text]):
        candidate = candidate.strip()
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}
        except json.JSONDecodeError:
            pass

    # Last resort: the outermost braces in the response.
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        try:
            parsed = json.loads(text[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return None
