"""
The advisor itself: one class that can be run as any of the four variants,
so the comparison in Phase 5 is genuinely like-for-like (same question, same
model, same temperature, same parser -- only the prompt ingredients differ).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from advisor import config, conflicts, prompts
from advisor.eligibility import EligibilityEngine, StudentRecord
from advisor.llm import LLMClient, LLMResponse
from advisor.retriever import HybridRetriever, Hit

_CODE_RE = re.compile(r"\b([A-Z]{3,5})\s?-?\s?(\d{3}[A-Z]?)\b")


@dataclass
class AdvisorAnswer:
    question: str
    variant: str
    text: str
    citations: list[str] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    latency_s: float = 0.0
    abstained: bool = False
    asked_followup: bool = False
    confidence: str = ""
    engine_facts: str = ""
    ok: bool = True
    error: str = ""
    provider: str = ""
    model: str = ""
    fallback_used: bool = False

    @property
    def answer_section(self) -> str:
        m = re.search(r"ANSWER:\s*(.+?)(?:\n[A-Z][A-Z \-]+:|\Z)", self.text, re.S)
        return (m.group(1) if m else self.text).strip()


class AcademicAdvisor:
    def __init__(self, variant: str = "V4_rag_student",
                 retriever: HybridRetriever | None = None,
                 engine: EligibilityEngine | None = None,
                 llm: LLMClient | None = None):
        assert variant in prompts.VARIANTS, f"unknown variant {variant}"
        self.variant = variant
        self.llm = llm or LLMClient()
        self.uses_rag = variant in ("V3_rag", "V4_rag_student")
        self.uses_student = variant == "V4_rag_student"
        self.retriever = retriever if retriever is not None else (
            HybridRetriever() if self.uses_rag else None)
        self.engine = engine if engine is not None else (
            EligibilityEngine() if self.uses_student else None)
        self.last_conflicts: list = []

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def mentioned_courses(question: str, engine: EligibilityEngine | None) -> list[str]:
        """Course codes written explicitly, plus titles matched against the catalogue."""
        found = [f"{a}{b}" for a, b in _CODE_RE.findall(question.upper())]
        if engine is not None:
            q = question.upper()
            for title, code in zip(engine.courses.title, engine.courses.course_code):
                t = str(title).upper()
                if len(t) > 8 and t in q and code not in found:
                    found.append(code)
        return list(dict.fromkeys(found))[:4]

    @staticmethod
    def parse_citations(text: str) -> list[str]:
        return list(dict.fromkeys(re.findall(r"\[([^\[\]]{6,160})\]", text)))

    @staticmethod
    def _confidence(text: str) -> str:
        m = re.search(r"CONFIDENCE:\s*(High|Medium|Low)", text, re.I)
        return m.group(1).title() if m else ""

    @staticmethod
    def _asked_followup(text: str) -> bool:
        m = re.search(r"FOLLOW-?UP:\s*(.+?)(?:\n[A-Z][A-Z \-]+:|\Z)", text, re.S | re.I)
        if not m:
            return "?" in text.split("ANSWER:")[-1][:400]
        body = m.group(1).strip()
        return bool(body) and body.lower().rstrip(".") not in {"none", "n/a", "-"}

    # -- main entry point --------------------------------------------------
    def answer(self, question: str, student: StudentRecord | None = None) -> AdvisorAnswer:
        t0 = time.perf_counter()
        hits: list[Hit] = []
        evidence = student_block = ""

        if self.uses_rag:
            # metadata filter: only this student's curriculum batch is eligible
            hits = self.retriever.search(question, k=config.TOP_K,
                                         batch=(student.batch if student else None))
            if self.retriever.is_out_of_corpus(hits):
                # Abstain WITHOUT calling the model: the corpus demonstrably
                # does not cover this, so a generation step can only hallucinate.
                return AdvisorAnswer(
                    question, self.variant, prompts.NO_EVIDENCE_MESSAGE,
                    citations=[], hits=hits, latency_s=time.perf_counter() - t0,
                    abstained=True, confidence="High")
            evidence = "\n\n---\n\n".join(h.as_evidence() for h in hits)

        if self.uses_student:
            codes = self.mentioned_courses(question, self.engine)
            blocks = []
            if student is not None:
                blocks.append(self.engine.profile_brief(student, mentioned=codes))
            # Surface real contradictions in the curriculum for the courses asked
            # about, so the advisor flags them instead of picking a side silently.
            if codes:
                found = conflicts.for_courses(
                    codes, batch=(student.batch if student else None),
                    courses=self.engine.courses)
                cb = conflicts.to_prompt_block(found)
                if cb:
                    blocks.append(cb)
                    self.last_conflicts = found
            student_block = "\n\n".join(blocks)

        user = prompts.build_user_message(question, self.variant, evidence, student_block)
        resp: LLMResponse = self.llm.generate(prompts.SYSTEMS[self.variant], user)

        if not resp.ok:
            return AdvisorAnswer(question, self.variant, "", hits=hits,
                                 latency_s=time.perf_counter() - t0,
                                 ok=False, error=resp.error, engine_facts=student_block)

        text = resp.text
        return AdvisorAnswer(
            question=question,
            variant=self.variant,
            text=text,
            citations=self.parse_citations(text),
            hits=hits,
            latency_s=time.perf_counter() - t0 - resp.wait_s,
            abstained="INSUFFICIENT INFORMATION" in text.strip().upper()[:120],
            asked_followup=self._asked_followup(text),
            confidence=self._confidence(text),
            engine_facts=student_block,
            provider=resp.provider,
            model=resp.model,
            fallback_used=resp.fallback_used,
        )


def build_all_variants(api_key: str | None = None,
                       max_wait: float | None = None) -> dict[str, AcademicAdvisor]:
    """Share one retriever/engine/LLM across variants so timings are comparable."""
    llm = LLMClient(api_key=api_key, max_wait=max_wait)
    retr = HybridRetriever()
    eng = EligibilityEngine()
    out = {}
    for v in prompts.VARIANTS:
        out[v] = AcademicAdvisor(
            v,
            retriever=retr if v in ("V3_rag", "V4_rag_student") else None,
            engine=eng if v == "V4_rag_student" else None,
            llm=llm,
        )
    return out
