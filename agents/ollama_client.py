# file: agents/ollama_client.py
"""
Thin wrapper around Ollama's /api/generate endpoint.
Handles timeouts, qwen3 thinking-block stripping, and JSON extraction.

qwen3 thinking control:
  - Ollama 0.6.0+: passes "think": false in options to suppress thinking entirely
  - Fallback: _strip_thinking() removes the block from raw output for older versions
"""

from __future__ import annotations
import json
import re
import httpx

try:
    from config import OLLAMA_BASE_URL, OLLAMA_MODEL, OLLAMA_TIMEOUT
except ImportError:
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL    = "qwen3:14b-q4_K_M"
    OLLAMA_TIMEOUT  = 120


# ── Thinking block stripper (fallback for Ollama < 0.6) ──────────────────────

def _strip_thinking(text: str) -> str:
    """
    Strip qwen3 extended-thinking blocks from raw model output.

    qwen3 emits reasoning in one of these forms before the real answer:

      Form A — standard XML tags:
        <think>...</think>  ## Real answer here

      Form B — bracket wrapper:
        [ long reasoning... ]
        ## Real answer here

      Form C — hybrid:
        [</think>
          actual answer
        ]
    """
    # Forms A and C both contain </think>
    idx = text.find("</think>")
    if idx != -1:
        after = text[idx + len("</think>"):].strip()
        # Form C: answer is wrapped in [...] after </think> — unwrap
        if after.startswith("["):
            after = after[1:]
        if after.endswith("]"):
            after = after[:-1]
        return after.strip()

    # Form B — large bracket block at start, real content follows after
    if text.lstrip().startswith("["):
        depth = 0
        start = text.index("[")
        close_pos = -1
        for i, ch in enumerate(text[start:], start):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    close_pos = i
                    break
        if close_pos != -1 and (close_pos - start) > 200:
            remainder = text[close_pos + 1:].strip()
            if remainder:
                return remainder

    return text.strip()


# ── Ollama version detection ──────────────────────────────────────────────────

_ollama_version: tuple[int, ...] | None = None

def _get_ollama_version() -> tuple[int, ...]:
    global _ollama_version
    if _ollama_version is not None:
        return _ollama_version
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/version", timeout=5)
        ver_str = r.json().get("version", "0.0.0")
        _ollama_version = tuple(int(x) for x in ver_str.split(".")[:3])
    except Exception:
        _ollama_version = (0, 0, 0)
    return _ollama_version


def _supports_think_param() -> bool:
    return _get_ollama_version() >= (0, 6, 0)


# ── Main call ─────────────────────────────────────────────────────────────────

def call_ollama(prompt: str,
                system: str = "",
                model: str | None = None,
                temperature: float = 0.2,
                stream: bool = False,
                think: bool = False) -> str:
    """
    Send a prompt to Ollama and return clean text.
    think=False (default) suppresses qwen3 extended thinking.
    """
    m = model or OLLAMA_MODEL
    payload: dict = {
        "model":  m,
        "prompt": prompt,
        "system": system,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }

    # Native suppression on Ollama 0.6+ — cleaner than post-processing
    if not think and _supports_think_param():
        payload["think"] = False

    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            r = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            raw = r.json().get("response", "").strip()
            return _strip_thinking(raw)   # no-op if think:false already worked
    except httpx.ConnectError:
        return "[ERROR] Cannot connect to Ollama. Is it running? → ollama serve"
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"


# ── JSON extraction ───────────────────────────────────────────────────────────

def extract_json(text: str) -> dict | list | None:
    """Extract JSON object or array from model output, handling fences/prose."""
    text = _strip_thinking(text)
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for pattern in [r"\[.*\]", r"\{.*\}"]:
        found = re.search(pattern, text, re.DOTALL)
        if found:
            try:
                return json.loads(found.group())
            except json.JSONDecodeError:
                pass
    return None


# ── Connection check ──────────────────────────────────────────────────────────

def check_ollama_connection() -> tuple[bool, str]:
    """Returns (ok, message)."""
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models  = [m["name"] for m in r.json().get("models", [])]
        ver     = _get_ollama_version()
        ver_str = ".".join(str(x) for x in ver) if ver != (0, 0, 0) else "unknown"
        think_s = "think:false ✓" if _supports_think_param() else "think:false ✗ (upgrade to 0.6+)"
        if not models:
            return False, f"Ollama {ver_str} running but no models pulled."
        return True, f"Ollama {ver_str} — {', '.join(models)}  |  {think_s}"
    except Exception as e:
        return False, f"Ollama not reachable: {e}"