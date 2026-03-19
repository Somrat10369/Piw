# file: agents/ollama_client.py
"""
Thin wrapper around Ollama's /api/generate endpoint.
Handles timeouts and JSON extraction from model output.
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


def call_ollama(prompt: str,
                system: str = "",
                model: str | None = None,
                temperature: float = 0.2,
                stream: bool = False) -> str:
    """
    Send a prompt to Ollama and return the text response.
    temperature=0.2 keeps outputs deterministic and analytical.
    """
    m = model or OLLAMA_MODEL
    payload = {
        "model": m,
        "prompt": prompt,
        "system": system,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
        },
    }
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            r = client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("response", "").strip()
    except httpx.ConnectError:
        return "[ERROR] Cannot connect to Ollama. Is it running? → ollama serve"
    except Exception as e:
        return f"[ERROR] Ollama call failed: {e}"


def extract_json(text: str) -> dict | list | None:
    """
    Extract a JSON object or array from model output.
    Models sometimes wrap JSON in markdown fences.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    # Try whole string first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find first {...} or [...]
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def check_ollama_connection() -> tuple[bool, str]:
    """Returns (ok, message). Check this before starting the pipeline."""
    try:
        r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not models:
            return False, "Ollama is running but no models are pulled."
        return True, f"Ollama OK — available models: {', '.join(models)}"
    except Exception as e:
        return False, f"Ollama not reachable: {e}"