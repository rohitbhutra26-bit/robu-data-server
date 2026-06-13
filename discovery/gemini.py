"""Tiny Gemini client for the Discovery agents.

Model durability: model names get deprecated often (gemini-1.5-flash and even
2.0-flash are retired by mid-2026). So instead of hardcoding one, we try a
chain of current models and use the first that responds. Override with the
GEMINI_MODEL env var if you want to pin a specific one.

Returns parsed JSON or None on any failure, so callers fall back gracefully.
"""

from __future__ import annotations
from typing import Optional
import os
import json
import re
import requests

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

# Tried in order; first one that returns 200 wins. 'latest' aliases track the
# current stable flash model and survive most deprecations.
_MODEL_CHAIN = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

# Cache which model worked so we don't re-probe dead ones every call.
_WORKING_MODEL: Optional[str] = None


def _models() -> list:
    env = os.environ.get("GEMINI_MODEL", "").strip()
    chain = ([env] + _MODEL_CHAIN) if env else list(_MODEL_CHAIN)
    if _WORKING_MODEL:
        chain = [_WORKING_MODEL] + [m for m in chain if m != _WORKING_MODEL]
    # de-dup preserving order
    seen, out = set(), []
    for m in chain:
        if m not in seen:
            seen.add(m); out.append(m)
    return out


def gemini_json(prompt: str, api_key: str, *, temperature: float = 0.25,
                max_tokens: int = 1100, timeout: int = 30) -> Optional[dict]:
    global _WORKING_MODEL
    if not api_key:
        return None

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }

    last_err = ""
    for model in _models():
        try:
            resp = requests.post(
                f"{_BASE}/{model}:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload, timeout=timeout,
            )
            if resp.status_code in (404, 400) and "model" in resp.text.lower():
                last_err = f"{model}: {resp.status_code}"
                continue  # model unavailable — try the next in the chain
            if not resp.ok:
                last_err = f"{model}: HTTP {resp.status_code} {resp.text[:160]}"
                continue
            data = resp.json()
            raw = (data.get("candidates", [{}])[0]
                       .get("content", {})
                       .get("parts", [{}])[0]
                       .get("text", ""))
            parsed = _extract_json(raw)
            if parsed is not None:
                _WORKING_MODEL = model
                return parsed
            last_err = f"{model}: unparseable response"
        except Exception as e:
            last_err = f"{model}: {e}"
            continue

    print(f"[discovery.gemini] all models failed — {last_err}")
    return None


def _extract_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None
