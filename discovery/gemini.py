"""Tiny Gemini client (mirrors the pattern in ai-analysis/route.ts).

Uses gemini-1.5-flash via REST. Returns parsed JSON or None on any failure,
so callers can fall back gracefully.
"""

from __future__ import annotations
from typing import Optional, Any
import json
import re
import requests

_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"


def gemini_json(prompt: str, api_key: str, *, temperature: float = 0.2,
                max_tokens: int = 1100, timeout: int = 30) -> Optional[dict]:
    if not api_key:
        return None
    try:
        resp = requests.post(
            f"{_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json",
                },
            },
            timeout=timeout,
        )
        if not resp.ok:
            print(f"[discovery.gemini] HTTP {resp.status_code}: {resp.text[:200]}")
            return None
        data = resp.json()
        raw = (data.get("candidates", [{}])[0]
                   .get("content", {})
                   .get("parts", [{}])[0]
                   .get("text", ""))
        return _extract_json(raw)
    except Exception as e:
        print(f"[discovery.gemini] {e}")
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
