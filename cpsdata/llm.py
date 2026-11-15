"""Minimal client for the free Google Gemini API (OpenAI-compatible endpoint).

The key is read from GEMINI_API_KEY (environment or a .env file in the repo
root or in ../seeder-app).  When a model's free daily quota is exhausted the
client moves on to the next model in ``models``.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from .catalog import ROOT

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3-flash-preview", "gemini-3.5-flash-lite",
                  "gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview"]


def load_env() -> None:
    for path in (ROOT / ".env", ROOT.parent / "seeder-app" / ".env"):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class QuotaExhausted(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, models: Optional[List[str]] = None, min_interval: float = 6.5,
                 temperature: float = 0.8, max_tokens: int = 8192,
                 reasoning_effort: Optional[str] = "low"):
        load_env()
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set. Get a free key at https://aistudio.google.com/apikey "
                             "and add GEMINI_API_KEY=... to .env")
        self.client = OpenAI(api_key=key, base_url=GEMINI_BASE_URL)
        self.models = list(models or DEFAULT_MODELS)
        self.min_interval = min_interval
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self._last = 0.0

    @property
    def model(self) -> str:
        if not self.models:
            raise QuotaExhausted("free quota of all models is exhausted; run again tomorrow to resume")
        return self.models[0]

    def chat(self, system: str, user: str, json_mode: bool = True, retries: int = 6) -> str:
        delay = 10.0
        for attempt in range(retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                kwargs = dict(model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                              messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                if self.reasoning_effort:
                    kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content or ""
                if text.strip():
                    return text
                raise RuntimeError("empty response")
            except QuotaExhausted:
                raise
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                low = msg.lower()
                if "429" in msg or "resource_exhausted" in low or "quota" in low:
                    if "perday" in low.replace(" ", "").replace("_", "") or "per day" in low or "daily" in low:
                        print(f"  daily quota of {self.model} exhausted; switching model")
                        self.models.pop(0)
                        self.model  # raises when none left
                        continue
                    m = re.search(r"retry in ([\d.]+)s", low) or re.search(r"retrydelay\W+(\d+)", low)
                    sleep = float(m.group(1)) + 1 if m else delay
                    print(f"  rate limited; sleeping {sleep:.0f}s")
                    time.sleep(sleep)
                    delay = min(delay * 2, 120)
                    continue
                if "404" in msg or "not found" in low or "no longer available" in low:
                    print(f"  model {self.model} unavailable; switching model")
                    self.models.pop(0)
                    self.model
                    continue
                if "503" in msg or "unavailable" in low or "high demand" in low:
                    print(f"  {self.model} overloaded; sleeping {delay:.0f}s")
                    time.sleep(delay)
                    delay = min(delay * 2, 120)
                    continue
                if attempt == retries - 1:
                    raise
                print(f"  error ({msg[:120]}); retrying in {delay:.0f}s")
                time.sleep(delay)
                delay = min(delay * 2, 120)
        raise RuntimeError("no response after retries")


def parse_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None
