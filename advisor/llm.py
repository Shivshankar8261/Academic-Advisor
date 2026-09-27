"""
Thin, provider-agnostic LLM wrapper.

Only three things are needed by the rest of the system:
  * deterministic generation (temperature 0) so the evaluation is reproducible
  * wall-clock latency, because response time is a graded metric
  * graceful failure, so a quota error is recorded as a failed test case
    rather than crashing a 30-case evaluation run half way through.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from advisor import config


@dataclass
class LLMResponse:
    text: str
    latency_s: float
    model: str
    ok: bool = True
    error: str = ""
    usage: dict = field(default_factory=dict)


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None,
                 api_key: str | None = None, temperature: float | None = None):
        self.provider = provider or config.LLM_PROVIDER
        self.model = model or config.GEMINI_MODEL
        self.temperature = config.TEMPERATURE if temperature is None else temperature
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, system: str, user: str, max_retries: int = 3) -> LLMResponse:
        if not self.available:
            return LLMResponse("", 0.0, self.model, ok=False,
                               error="No GOOGLE_API_KEY / GEMINI_API_KEY set.")
        t0 = time.perf_counter()
        last = ""
        for attempt in range(max_retries):
            try:
                resp = self._get().models.generate_content(
                    model=self.model,
                    contents=user,
                    config={
                        "system_instruction": system,
                        "temperature": self.temperature,
                        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
                    },
                )
                dt = time.perf_counter() - t0
                usage = {}
                if getattr(resp, "usage_metadata", None):
                    u = resp.usage_metadata
                    usage = {
                        "prompt_tokens": getattr(u, "prompt_token_count", None),
                        "output_tokens": getattr(u, "candidates_token_count", None),
                    }
                return LLMResponse((resp.text or "").strip(), dt, self.model, usage=usage)
            except Exception as exc:                      # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt * 1.5)        # backoff for 429s
        return LLMResponse("", time.perf_counter() - t0, self.model, ok=False, error=last)


class EchoLLM(LLMClient):
    """Offline stand-in so the pipeline and harness can be tested without a key."""

    def __init__(self, canned: str = "[offline stub]"):
        super().__init__()
        self.canned = canned

    @property
    def available(self) -> bool:
        return True

    def generate(self, system: str, user: str, max_retries: int = 3) -> LLMResponse:
        time.sleep(0.01)
        return LLMResponse(self.canned, 0.01, "echo-stub")


if __name__ == "__main__":
    c = LLMClient()
    print("key present:", c.available, "| model:", c.model)
    if c.available:
        r = c.generate("You are terse.", "Reply with exactly: OK")
        print(r.ok, repr(r.text), f"{r.latency_s:.2f}s", r.usage)
