"""
Phase 4 -- evaluation methodology and scoring.

Every response is classified into exactly one of four buckets, using
deterministic rules so the scoring is reproducible and auditable rather than
a matter of opinion:

  correct            did the right thing AND contained every required fact
  partially_correct  did the right thing but missed some required facts
  unsupported        asserted content where the corpus has none (hallucination)
  incorrect          contradicted a verified fact, or took the wrong action

Two distinctions the assignment explicitly asks for:
  Correctness  -- does the answer match the verified rule?      -> accuracy
  Reliability  -- does it stay grounded, admit gaps, avoid
                  unsupported recommendations?                  -> hallucination
                                                                   rate, missing-info
                                                                   and conflict handling
A system can be correct on easy questions and unreliable on ambiguous ones,
which is exactly why both are reported separately.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, asdict, field

import pandas as pd

from advisor import config
from advisor.advisor import AcademicAdvisor, AdvisorAnswer
from advisor.eligibility import StudentRecord
from advisor.testset import TestCase, ANSWER, ABSTAIN, ASK, FLAG

CORRECT, PARTIAL, UNSUPPORTED, INCORRECT, ERROR = (
    "correct", "partially_correct", "unsupported", "incorrect", "error")

_CONFLICT_WORDS = re.compile(
    r"conflict|contradict|discrepanc|inconsistent|cannot be satisfied|"
    r"differs? (by|between|depending)|co-?requisite|not offered anywhere", re.I)


def observed_behaviour(ans: AdvisorAnswer) -> str:
    if ans.abstained:
        return ABSTAIN
    if _CONFLICT_WORDS.search(ans.text):
        return FLAG
    if ans.asked_followup:
        return ASK
    return ANSWER


def _behaviour_ok(expected: str, got: str) -> bool:
    if expected == got:
        return True
    # A conflict may legitimately be handled either by naming the conflict or by
    # asking which side applies; both are non-committal and both are acceptable.
    if expected == FLAG and got in (ASK, FLAG):
        return True
    if expected == ASK and got == FLAG:
        return True
    return False


def _source_ok(case: TestCase, ans: AdvisorAnswer) -> bool | None:
    """None when the case has no attributable source (abstentions, out-of-scope)."""
    exp = case.expected_source
    if exp.lower().startswith("no source") or "not extractable" in exp.lower():
        return None
    blob = " ".join(ans.citations) + " " + ans.text
    m = re.search(r"Clause\s*([\d.]+)", exp)
    if m:
        want = m.group(1).rstrip(".")
        return _clause_cited(want, blob)
    for key in ("Programme Structure", "Semester Spread", "Student Handbook",
                "SOP", "Minor Courses"):
        if key.lower() in exp.lower():
            return key.lower() in blob.lower()
    return None


def _clause_key(c: str) -> tuple:
    return tuple(int(x) for x in c.strip(".").split(".") if x.isdigit())


def _clause_cited(want: str, blob: str) -> bool:
    """True if the expected clause is cited exactly, or falls inside a cited
    clause range ('Clause 7-7.2' covers 7.2), or its parent clause is cited."""
    w = _clause_key(want)
    for a, b in re.findall(r"Clause[s]?\s*(\d+(?:\.\d+)*)(?:\s*[–-]\s*(\d+(?:\.\d+)*))?", blob, re.I):
        lo, hi = _clause_key(a), _clause_key(b or a)
        if not lo:
            continue
        if lo == w or (lo <= w <= hi) or (w[:len(lo)] == lo and not b):
            return True
    return False


@dataclass
class CaseResult:
    case_id: str
    category: str
    variant: str
    question: str
    behaviour_expected: str
    behaviour_observed: str
    behaviour_ok: bool
    label: str
    matched: int
    required: int
    traps_hit: list[str] = field(default_factory=list)
    source_ok: bool | None = None
    latency_s: float = 0.0
    citations: int = 0
    answer: str = ""
    error: str = ""
    provider: str = ""
    model: str = ""
    top_rerank: float | None = None


def score_case(case: TestCase, ans: AdvisorAnswer) -> CaseResult:
    if not ans.ok:
        return CaseResult(case.id, case.category, ans.variant, case.question,
                          case.behaviour, "ERROR", False, ERROR, 0,
                          len(case.must_include), latency_s=ans.latency_s,
                          error=ans.error)

    text = ans.text
    got = observed_behaviour(ans)
    b_ok = _behaviour_ok(case.behaviour, got)

    matched = [p for p in case.must_include if re.search(p, text, re.I)]
    traps = [p for p in case.must_not_include if re.search(p, text, re.I | re.M)]

    if traps:
        # A trap is a plausible-but-wrong value. Asserting one where the corpus
        # is silent is a hallucination; asserting one against a verified rule is
        # simply incorrect.
        label = UNSUPPORTED if case.behaviour == ABSTAIN else INCORRECT
    elif not b_ok:
        if case.behaviour == ABSTAIN:
            label = UNSUPPORTED           # answered where nothing supports an answer
        else:
            label = INCORRECT             # wrong action (over-abstained, or advised blindly)
    elif not case.must_include:
        label = CORRECT
    elif len(matched) == len(case.must_include):
        label = CORRECT
    elif matched:
        label = PARTIAL
    else:
        label = INCORRECT

    return CaseResult(case.id, case.category, ans.variant, case.question,
                      case.behaviour, got, b_ok, label, len(matched),
                      len(case.must_include), traps,
                      _source_ok(case, ans), ans.latency_s, len(ans.citations),
                      text[:1500], "", ans.provider or ("none" if ans.abstained else ""),
                      ans.model, (max(h.score for h in ans.hits) if ans.hits else None))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_variant(advisor: AcademicAdvisor, cases: list[TestCase],
                students: dict[str, StudentRecord], sleep: float = 0.0,
                progress: bool = True) -> list[CaseResult]:
    out = []
    for i, case in enumerate(cases, 1):
        stu = students.get(case.student_id) if case.student_id else None
        ans = advisor.answer(case.question, student=stu)
        res = score_case(case, ans)
        out.append(res)
        if progress:
            flag = {CORRECT: "OK ", PARTIAL: "~  ", UNSUPPORTED: "HAL",
                    INCORRECT: "ERR", ERROR: "!!!"}[res.label]
            print(f"  [{i:2d}/{len(cases)}] {flag} {case.id} {res.label:<18} "
                  f"{res.latency_s:5.2f}s")
        if sleep:
            time.sleep(sleep)
    return out


# --------------------------------------------------------------------------
# Metrics -- the eight the assignment names
# --------------------------------------------------------------------------

ELIGIBILITY_CATS = {"prerequisite_eligibility", "credits_progression", "course_offering"}


def metrics(results: list[CaseResult]) -> dict:
    df = pd.DataFrame([asdict(r) for r in results])
    n = len(df)
    if n == 0:
        return {}
    lab = df.label

    def pct(mask) -> float:
        return round(100.0 * mask.sum() / max(1, n), 1)

    elig = df[df.category.isin(ELIGIBILITY_CATS)]
    miss = df[df.category.isin({"missing_information", "out_of_scope"})]
    conf = df[df.category == "conflicting_rules"]
    src = df[df.source_ok.notna()]

    return {
        "n_cases": n,
        # 1. Accuracy -- full credit only; partial reported separately
        "accuracy_%": pct(lab == CORRECT),
        "accuracy_incl_partial_%": pct(lab.isin([CORRECT, PARTIAL])),
        # 2. Hallucination rate
        "hallucination_rate_%": pct(lab == UNSUPPORTED),
        # 3. Correct eligibility decisions
        "eligibility_correct_%": (round(100.0 * (elig.label == CORRECT).sum() / len(elig), 1)
                                  if len(elig) else None),
        # 4. Source / evidence correctness
        "source_correct_%": (round(100.0 * src.source_ok.sum() / len(src), 1)
                             if len(src) else None),
        # 5. Response time
        "mean_latency_s": round(float(df.latency_s.mean()), 2),
        "p90_latency_s": round(float(df.latency_s.quantile(0.9)), 2),
        # 6. Handling of missing information
        "missing_info_handled_%": (round(100.0 * miss.behaviour_ok.sum() / len(miss), 1)
                                   if len(miss) else None),
        # 7. Handling of conflicting rules
        "conflict_handled_%": (round(100.0 * conf.behaviour_ok.sum() / len(conf), 1)
                               if len(conf) else None),
        # 8. Incorrect recommendations
        "incorrect_recommendations": int((lab == INCORRECT).sum()),
        # supporting detail
        "partially_correct": int((lab == PARTIAL).sum()),
        "errors": int((lab == ERROR).sum()),
        "mean_citations": round(float(df.citations.mean()), 2),
        "followup_rate_%": pct(df.behaviour_observed == ASK),
        "fallback_answers": int((df.provider == "gemini").sum()) if "provider" in df else 0,
        "abstention_rate_%": pct(df.behaviour_observed == ABSTAIN),
    }


def comparison_table(all_results: dict[str, list[CaseResult]]) -> pd.DataFrame:
    from advisor.prompts import VARIANT_LABELS
    rows = []
    for v, res in all_results.items():
        m = metrics(res)
        m["variant"] = VARIANT_LABELS.get(v, v)
        rows.append(m)
    df = pd.DataFrame(rows).set_index("variant")
    front = ["accuracy_%", "accuracy_incl_partial_%", "hallucination_rate_%",
             "eligibility_correct_%", "source_correct_%", "mean_latency_s",
             "missing_info_handled_%", "conflict_handled_%", "incorrect_recommendations"]
    return df[front + [c for c in df.columns if c not in front]]


def per_category(all_results: dict[str, list[CaseResult]]) -> pd.DataFrame:
    from advisor.prompts import VARIANT_LABELS
    rows = []
    for v, res in all_results.items():
        df = pd.DataFrame([asdict(r) for r in res])
        for cat, g in df.groupby("category"):
            rows.append({"variant": VARIANT_LABELS.get(v, v), "category": cat,
                         "n": len(g),
                         "correct_%": round(100.0 * (g.label == CORRECT).sum() / len(g), 1)})
    return (pd.DataFrame(rows)
            .pivot(index="category", columns="variant", values="correct_%")
            .fillna(0.0))


def save_results(all_results: dict[str, list[CaseResult]], tag: str = "") -> dict:
    suffix = f"_{tag}" if tag else ""
    raw = pd.DataFrame([asdict(r) for res in all_results.values() for r in res])
    paths = {
        "raw": config.RESULTS / f"raw_results{suffix}.csv",
        "metrics": config.RESULTS / f"metrics{suffix}.csv",
        "per_category": config.RESULTS / f"per_category{suffix}.csv",
    }
    raw.to_csv(paths["raw"], index=False)
    comparison_table(all_results).to_csv(paths["metrics"])
    per_category(all_results).to_csv(paths["per_category"])
    return paths
