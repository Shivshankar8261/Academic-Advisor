"""
LLM router: Groq primary, Gemini fallback.

    Groq  openai/gpt-oss-120b   fast, strong, free tier = 8,000 tokens / minute
      │   (then openai/gpt-oss-20b, which has its own separate quota)
      │   on rate-limit: wait for the minute window if it is short enough,
      │   otherwise, or on any other failure ...
      ▼
    Gemini gemini-3.6-flash     tried with each configured key in turn; a key
                                that is refused (403) is disabled for the session

Every response records which provider and model produced it, so the
evaluation can report exactly how many answers came from the fallback.

A token-bucket throttle tracks Groq usage over a rolling 60 s window and waits
BEFORE sending, rather than firing requests into a 429.
"""
from __future__ import annotations

import re
import time
from collections import deque
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
    provider: str = ""
    fallback_used: bool = False
    wait_s: float = 0.0     # time spent waiting on rate limits (excluded from latency)


class _TokenWindow:
    """Rolling 60-second token counter for a per-minute quota."""

    def __init__(self, limit: int):
        self.limit = limit
        self.events: deque[tuple[float, int]] = deque()

    def _used(self, now: float) -> int:
        while self.events and now - self.events[0][0] > 60:
            self.events.popleft()
        return sum(n for _, n in self.events)

    def wait_needed(self, want: int) -> float:
        now = time.time()
        used = self._used(now)
        if used + want <= self.limit or not self.events:
            return 0.0
        # wait until enough old usage has aged out of the window
        need = used + want - self.limit
        freed = 0
        for t, n in self.events:
            freed += n
            if freed >= need:
                return max(0.0, 60 - (now - t) + 0.5)
        return 60.0

    def add(self, n: int) -> None:
        self.events.append((time.time(), n))


def _estimate_tokens(*texts: str) -> int:
    return int(sum(len(t) for t in texts) / 3.6) + 600   # + output/reasoning allowance


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str, model: str):
        from groq import Groq
        self.client = Groq(api_key=api_key, max_retries=0)
        self.model = model
        self.window = _TokenWindow(config.GROQ_TPM)
        self.exhausted = False          # daily quota gone -> skip for the session

    def generate(self, system: str, user: str, max_wait: float) -> LLMResponse:
        if self.exhausted:
            return LLMResponse("", 0.0, self.model, ok=False, provider=self.name,
                               error="daily token quota exhausted")
        want = _estimate_tokens(system, user)
        wait = self.window.wait_needed(want)
        waited = 0.0
        if wait > max_wait:
            return LLMResponse("", 0.0, self.model, ok=False, provider=self.name,
                               error=f"rate window needs {wait:.0f}s > max_wait {max_wait:.0f}s")
        if wait:
            time.sleep(wait)
            waited += wait
        kwargs = {}
        if "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = config.GROQ_REASONING_EFFORT
        t0 = time.perf_counter()
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                    temperature=config.TEMPERATURE,
                    max_tokens=config.MAX_OUTPUT_TOKENS, **kwargs)
                used = r.usage.total_tokens if r.usage else want
                self.window.add(used)
                text = (r.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError("empty completion (reasoning consumed the budget)")
                return LLMResponse(text, time.perf_counter() - t0 - (waited - (wait or 0)),
                                   self.model,
                                   usage={"prompt_tokens": r.usage.prompt_tokens,
                                          "output_tokens": r.usage.completion_tokens},
                                   provider=f"{self.name}", wait_s=waited)
            except Exception as exc:                          # noqa: BLE001
                msg = str(exc)
                if "per day" in msg.lower() or "(TPD)" in msg:
                    self.exhausted = True
                    return LLMResponse("", time.perf_counter() - t0, self.model, ok=False,
                                       provider=self.name, error="daily token quota exhausted")
                if "429" in msg or "rate_limit" in msg.lower():
                    m = re.search(r"try again in ([\d.]+)(ms|s)", msg)
                    delay = (float(m.group(1)) / (1000 if m.group(2) == "ms" else 1)
                             if m else 20.0) + 0.5
                    self.window.add(config.GROQ_TPM)          # treat window as full
                    if delay <= max_wait and attempt < 2:
                        time.sleep(delay)
                        waited += delay
                        continue
                return LLMResponse("", time.perf_counter() - t0, self.model, ok=False,
                                   provider=self.name, error=f"{type(exc).__name__}: {msg[:300]}")
        return LLMResponse("", time.perf_counter() - t0, self.model, ok=False,
                           provider=self.name, error="groq retries exhausted")


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_keys: list[str], model: str):
        from google import genai
        self.clients = [genai.Client(api_key=k) for k in api_keys]
        self.dead: set[int] = set()
        self.model = model

    def generate(self, system: str, user: str, max_wait: float = 0) -> LLMResponse:
        t0 = time.perf_counter()
        last = "no usable Gemini key"
        for i, client in enumerate(self.clients):
            if i in self.dead:
                continue
            try:
                r = client.models.generate_content(
                    model=self.model, contents=user,
                    config={"system_instruction": system,
                            "temperature": config.TEMPERATURE,
                            "max_output_tokens": config.MAX_OUTPUT_TOKENS})
                u = getattr(r, "usage_metadata", None)
                return LLMResponse((r.text or "").strip(), time.perf_counter() - t0,
                                   self.model, provider=self.name,
                                   usage={"prompt_tokens": getattr(u, "prompt_token_count", None),
                                          "output_tokens": getattr(u, "candidates_token_count", None)})
            except Exception as exc:                          # noqa: BLE001
                last = f"{type(exc).__name__}: {str(exc)[:200]}"
                if "403" in last or "PERMISSION_DENIED" in last or "API_KEY_INVALID" in last:
                    self.dead.add(i)                          # never retry a refused key
        return LLMResponse("", time.perf_counter() - t0, self.model, ok=False,
                           provider=self.name, error=last)


class LLMClient:
    """Public interface used by the advisor. Same signature as before."""

    def __init__(self, groq_key: str | None = None, gemini_keys: list[str] | None = None,
                 max_wait: float | None = None, api_key: str | None = None):
        groq_key = groq_key or config._secret("GROQ_API_KEY")
        if gemini_keys is None:
            gemini_keys = [k for k in (api_key, config._secret("GOOGLE_API_KEY"),
                                       config._secret("GOOGLE_API_KEY_2")) if k]
            gemini_keys = list(dict.fromkeys(gemini_keys))
        self.max_wait = config.GROQ_MAX_WAIT_S if max_wait is None else max_wait
        self.providers = []
        if groq_key:
            for m in [config.GROQ_MODEL] + [x for x in config.GROQ_FALLBACK_MODELS
                                             if x != config.GROQ_MODEL]:
                self.providers.append(GroqProvider(groq_key, m))
        if gemini_keys:
            self.providers.append(GeminiProvider(gemini_keys, config.GEMINI_MODEL))
        self.model = self.providers[0].model if self.providers else "none"
        self.temperature = config.TEMPERATURE

    @property
    def available(self) -> bool:
        return bool(self.providers)

    def generate(self, system: str, user: str, max_retries: int = 3) -> LLMResponse:
        if not self.providers:
            return LLMResponse("", 0.0, "none", ok=False,
                               error="No GROQ_API_KEY or GOOGLE_API_KEY configured.")
        errors = []
        t0 = time.perf_counter()
        for i, p in enumerate(self.providers):
            t_p = time.perf_counter()
            r = p.generate(system, user, self.max_wait)
            if r.ok:
                r.fallback_used = i > 0
                # latency = time since the request started, minus rate-limit waiting
                r.latency_s = (time.perf_counter() - t0) - r.wait_s
                return r
            errors.append(f"{p.name}: {r.error}")
        return LLMResponse("", time.perf_counter() - t0, self.model, ok=False,
                           error=" | ".join(errors))


class EchoLLM(LLMClient):
    """Offline stand-in so the pipeline and harness can be tested without a key."""

    def __init__(self, canned: str = "[offline stub]"):
        self.canned, self.providers, self.model = canned, [object()], "echo-stub"
        self.max_wait, self.temperature = 0, 0.0

    def generate(self, system: str, user: str, max_retries: int = 3) -> LLMResponse:
        return LLMResponse(self.canned, 0.01, "echo-stub", provider="echo")


if __name__ == "__main__":
    c = LLMClient()
    print("providers:", [f"{p.name}:{p.model}" for p in c.providers])
    r = c.generate("Be terse.", "Reply with exactly: READY")
    print(r.provider, r.model, repr(r.text), f"{r.latency_s:.2f}s", r.usage)
    g = LLMClient(groq_key="gsk_invalid")        # force the fallback path
    r = g.generate("Be terse.", "Reply with exactly: READY")
    print("fallback test ->", r.provider, repr(r.text), "fallback_used =", r.fallback_used)
