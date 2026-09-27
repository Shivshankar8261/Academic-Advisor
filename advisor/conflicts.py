"""
Automatic detection of conflicting / unsatisfiable rules in the curriculum.

Phase 2 asks the advisor to "identify conflicts" and Phase 4 grades "handling
of conflicting rules". Rather than inventing conflicts, this module mines the
real ones out of the university's own programme structure. Three kinds:

  DANGLING_PREREQ  a course requires a course that its own batch never offers
  ORDERING         a prerequisite is offered no earlier than the course itself
  CROSS_BATCH      the same course code has different prerequisites per batch

Detected conflicts are injected into the prompt so the advisor flags them
instead of confidently reciting one side of a contradiction.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, asdict

import pandas as pd

from advisor import config


@dataclass
class Conflict:
    kind: str
    course: str
    batch: str
    detail: str
    advice: str

    def line(self) -> str:
        return f"[{self.kind}] {self.course} ({self.batch}): {self.detail} -> {self.advice}"


def _load(courses: pd.DataFrame | None = None) -> pd.DataFrame:
    df = courses if courses is not None else pd.read_csv(config.COURSES_CSV)
    if df["prereqs"].dtype == object:
        df = df.copy()
        df["prereqs"] = df["prereqs"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else v)
    return df


def detect(courses: pd.DataFrame | None = None) -> list[Conflict]:
    df = _load(courses)
    out: list[Conflict] = []

    for batch, g in df.groupby("batch"):
        codes = set(g.course_code)
        sem = {r.course_code: r.sem_num for r in g.itertuples()}
        for r in g.itertuples():
            for p in r.prereqs:
                alts = [a for a in p.split("/") if a]
                if not any(a in codes for a in alts):
                    out.append(Conflict(
                        "DANGLING_PREREQ", r.course_code, str(batch),
                        f"lists prerequisite {p}, which does not appear anywhere in the "
                        f"{batch} semester spread",
                        "the prerequisite cannot be completed within this curriculum; "
                        "confirm with the Programme Chair whether it is waived, renamed, "
                        "or carried over from an earlier curriculum version"))
                elif p in sem and pd.notna(sem[p]) and pd.notna(r.sem_num) and sem[p] >= r.sem_num:
                    out.append(Conflict(
                        "ORDERING", r.course_code, str(batch),
                        f"is offered in S{r.sem_num:.0f} but its prerequisite {p} is offered "
                        f"in S{sem[p]:.0f}, so the sequence cannot be satisfied as published",
                        "treat as a co-requisite or seek Programme Chair approval"))

    for code, g in df.groupby("course_code"):
        variants = {tuple(sorted(x)) for x in g.prereqs}
        if len(variants) > 1:
            per = {str(r.batch): r.prereq_raw for r in g.itertuples()}
            out.append(Conflict(
                "CROSS_BATCH", code, ", ".join(per),
                "has different prerequisites depending on the curriculum batch: "
                + "; ".join(f"{b} requires {v}" for b, v in per.items()),
                "the student's batch must be established before any eligibility "
                "statement about this course can be made"))
    return out


def for_courses(codes: list[str], batch: str | None = None,
                courses: pd.DataFrame | None = None) -> list[Conflict]:
    """Conflicts relevant to the specific courses a question mentions."""
    codes = {c.upper() for c in codes}
    hits = []
    for c in detect(courses):
        if c.course.upper() in codes or any(k in c.course.upper() for k in codes):
            if batch is None or c.kind == "CROSS_BATCH" or str(batch) in c.batch:
                hits.append(c)
    return hits


def to_prompt_block(conflicts: list[Conflict]) -> str:
    if not conflicts:
        return ""
    body = "\n".join(f"  - {c.line()}" for c in conflicts)
    return ("=== DETECTED RULE CONFLICTS (verified against the programme structure) ===\n"
            f"{body}\n"
            "You MUST surface any conflict above that bears on the question, state both "
            "sides, and say what the student should do rather than picking one silently.")


def save_report(path=None) -> pd.DataFrame:
    path = path or (config.RESULTS / "conflict_report.csv")
    df = pd.DataFrame([asdict(c) for c in detect()])
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    cs = detect()
    print(f"{len(cs)} conflicts detected")
    for k in ("CROSS_BATCH", "ORDERING", "DANGLING_PREREQ"):
        sub = [c for c in cs if c.kind == k]
        print(f"\n--- {k} ({len(sub)}) ---")
        for c in sub[:4]:
            print("  " + c.line()[:190])
    save_report()
