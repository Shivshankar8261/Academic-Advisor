"""
Deterministic academic rule engine.

The LLM is never asked to do arithmetic or to decide eligibility. Those are
computed here from the structured catalogue plus the student's record, and the
result is injected into the prompt as verified facts. This is what separates
system variant V4 (RAG + Structured Student Data) from V3 (RAG alone), and it
is why V4 is expected to win on the "correct eligibility decisions" metric.

Every rule carries the handbook clause it comes from, so the advisor can cite
a regulation for each verdict instead of asserting one.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

import pandas as pd

from advisor import config

# --- Rules lifted verbatim from the Student Handbook (August 2026) ---------
PASSING_GRADES = {"O", "A+", "A", "B+", "B", "C", "D", "S"}   # Clause 8.11
FAILING_GRADES = {"F", "FA", "U"}                             # Clause 8.13.3 / 8.13.4
MIN_ATTENDANCE = 75.0                                         # Clause 7.2
RELAXED_ATTENDANCE = 65.0                                     # Clause 7.3
PROGRESSION_CGPA = {2: 4.00, 3: 5.00, 4: 5.00}                # Clause 12.1, Table 3

RULE_SOURCE = {
    "passing":      "Student Handbook (August 2026) — Clause 8.11 (Earned Credits)",
    "fail_F":       "Student Handbook (August 2026) — Clause 8.13.3(c),(e)",
    "fail_FA":      "Student Handbook (August 2026) — Clause 8.13.3(g)",
    "attendance":   "Student Handbook (August 2026) — Clause 7.2 / 7.3",
    "progression":  "Student Handbook (August 2026) — Clause 12.1, Table 3",
    "prereq":       "Programme Structure & Semester Spread (Sept 2026)",
    "offering":     "Programme Structure & Semester Spread (Sept 2026)",
    "credits":      "Programme Structure & Semester Spread (Sept 2026) — basket minimums",
}

# NOTE: the letter-grade -> grade-point table (Table 2, Clause 8.10) is an
# IMAGE in the source PDF and does not survive text extraction. The engine
# therefore never computes a CGPA from letter grades; it only uses the CGPA
# that the student record states. Queries that require Table 2 are answered
# with an explicit "not available in the provided documents".
GRADE_POINTS_AVAILABLE = False


@dataclass
class StudentRecord:
    student_id: str
    batch: str
    programme: str
    current_semester: int
    cgpa: float | None
    completed: dict[str, str] = field(default_factory=dict)   # code -> grade
    attendance: dict[str, float] = field(default_factory=dict)  # code -> %
    declared_minor: str | None = None
    credits_earned_override: float | None = None
    notes: str = ""

    @property
    def academic_year(self) -> int:
        return (self.current_semester + 1) // 2

    def passed(self) -> set[str]:
        return {c for c, g in self.completed.items() if str(g).upper() in PASSING_GRADES}

    def failed(self) -> dict[str, str]:
        return {c: g for c, g in self.completed.items() if str(g).upper() in FAILING_GRADES}


@dataclass
class Verdict:
    question: str
    decision: str                 # ELIGIBLE | NOT_ELIGIBLE | CONDITIONAL | UNKNOWN
    reasons: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    facts: dict = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        lines = [f"DECISION: {self.decision}"]
        for r in self.reasons:
            lines.append(f"  - {r}")
        if self.sources:
            lines.append("  RULE SOURCES: " + "; ".join(dict.fromkeys(self.sources)))
        return "\n".join(lines)


class EligibilityEngine:
    def __init__(self, courses: pd.DataFrame | None = None,
                 baskets: pd.DataFrame | None = None):
        self.courses = courses if courses is not None else pd.read_csv(config.COURSES_CSV)
        self.baskets = baskets if baskets is not None else pd.read_csv(config.BASKETS_CSV)
        # prereqs round-trip through CSV as a string; restore the list
        if self.courses["prereqs"].dtype == object:
            self.courses["prereqs"] = self.courses["prereqs"].apply(
                lambda v: eval(v) if isinstance(v, str) and v.startswith("[") else []
            )

    # -- lookups ----------------------------------------------------------
    def find_course(self, query: str, batch: str | None = None) -> pd.DataFrame:
        q = str(query).strip().upper()
        df = self.courses
        if batch:
            df = df[df.batch.astype(str) == str(batch)]
            if df.empty:
                df = self.courses
        exact = df[df.course_code.str.upper().str.replace(" ", "") == q.replace(" ", "")]
        if not exact.empty:
            return exact
        contains = df[df.course_code.str.upper().str.contains(q.replace(" ", ""), regex=False)]
        if not contains.empty:
            return contains
        return df[df.title.str.upper().str.contains(q, regex=False, na=False)]

    def credits_of(self, code: str, batch: str | None = None) -> float | None:
        m = self.find_course(code, batch)
        return None if m.empty else float(m.iloc[0]["credits"])

    # -- individual checks ------------------------------------------------
    def check_prerequisites(self, student: StudentRecord, course: str) -> Verdict:
        m = self.find_course(course, student.batch)
        if m.empty:
            return Verdict(
                f"Prerequisites for '{course}'", "UNKNOWN",
                [f"'{course}' does not appear in the {student.batch} programme structure. "
                 f"It may not exist, may be spelled differently, or may belong to another batch."],
                [RULE_SOURCE["prereq"]],
            )
        row = m.iloc[0]
        needed = list(row["prereqs"])
        passed = student.passed()
        failed = student.failed()

        missing = [p for p in needed if p not in passed]
        # a slash-code prerequisite ("COMP210/COMP301") is satisfied by either
        resolved_missing = []
        for p in missing:
            alts = [a for a in p.split("/") if a]
            if not any(a in passed for a in alts):
                resolved_missing.append(p)

        reasons = [f"{row['course_code']} '{row['title']}' requires: "
                   f"{', '.join(needed) if needed else 'no prerequisites (NIL)'}."]
        if not needed:
            decision = "ELIGIBLE"
        elif not resolved_missing:
            decision = "ELIGIBLE"
            reasons.append(f"All prerequisites satisfied ({', '.join(needed)} passed).")
        else:
            decision = "NOT_ELIGIBLE"
            for p in resolved_missing:
                if p in failed:
                    reasons.append(
                        f"Prerequisite {p} was attempted but graded '{failed[p]}', which is "
                        f"not a passing grade, so it does not count as completed.")
                elif p in student.completed:
                    reasons.append(f"Prerequisite {p} is recorded but not passed.")
                else:
                    reasons.append(f"Prerequisite {p} has not been taken.")
        return Verdict(f"Prerequisites for {row['course_code']}", decision, reasons,
                       [RULE_SOURCE["prereq"], RULE_SOURCE["passing"]],
                       {"course": row["course_code"], "title": row["title"],
                        "required": needed, "missing": resolved_missing,
                        "credits": row["credits"]})

    def check_offered(self, course: str, semester: int, batch: str) -> Verdict:
        m = self.find_course(course, batch)
        if m.empty:
            return Verdict(f"Is '{course}' offered in S{semester}?", "UNKNOWN",
                           [f"'{course}' is not listed in the {batch} programme structure."],
                           [RULE_SOURCE["offering"]])
        sems = sorted({s for s in m["semester"].dropna() if s})
        target = f"S{semester}"
        ok = target in sems
        return Verdict(
            f"Is {m.iloc[0]['course_code']} offered in {target}?",
            "ELIGIBLE" if ok else "NOT_ELIGIBLE",
            [f"{m.iloc[0]['course_code']} is listed in semester(s): {', '.join(sems) or 'none listed'}."
             + ("" if ok else f" It is not listed for {target} in the {batch} spread.")],
            [RULE_SOURCE["offering"]],
            {"offered_in": sems, "target": target},
        )

    def check_attendance(self, student: StudentRecord, course: str | None = None) -> Verdict:
        att = student.attendance
        if not att:
            return Verdict("Attendance eligibility", "UNKNOWN",
                           ["No attendance data is recorded for this student."],
                           [RULE_SOURCE["attendance"]])
        items = {course: att[course]} if course and course in att else att
        short = {c: v for c, v in items.items() if v < MIN_ATTENDANCE}
        if not short:
            return Verdict("Attendance eligibility", "ELIGIBLE",
                           [f"All recorded courses meet the {MIN_ATTENDANCE:.0f}% minimum."],
                           [RULE_SOURCE["attendance"]], {"attendance": items})
        hard = {c: v for c, v in short.items() if v < RELAXED_ATTENDANCE}
        reasons = [f"{c}: {v:.0f}% — below the {MIN_ATTENDANCE:.0f}% requirement (Clause 7.2)."
                   for c, v in short.items()]
        if hard:
            reasons.append(
                f"{', '.join(hard)} also fall below the absolute floor of {RELAXED_ATTENDANCE:.0f}% "
                f"(Clause 7.3), which cannot be relaxed under any circumstances.")
            decision = "NOT_ELIGIBLE"
        else:
            reasons.append(
                f"Between {RELAXED_ATTENDANCE:.0f}% and {MIN_ATTENDANCE:.0f}%, relaxation is possible "
                f"only on documented medical/representation grounds with Programme Chair and "
                f"Vice-Chancellor approval (Clause 7.3 / 7.4).")
            decision = "CONDITIONAL"
        return Verdict("Attendance eligibility", decision, reasons,
                       [RULE_SOURCE["attendance"]], {"shortfall": short})

    def check_progression(self, student: StudentRecord) -> Verdict:
        target_year = student.academic_year + 1
        need = PROGRESSION_CGPA.get(target_year, PROGRESSION_CGPA[3])
        if student.cgpa is None:
            return Verdict(f"Progression to Year {target_year}", "UNKNOWN",
                           ["The student record does not contain a CGPA."],
                           [RULE_SOURCE["progression"]])
        ok = student.cgpa >= need
        return Verdict(
            f"Progression to Year {target_year}",
            "ELIGIBLE" if ok else "NOT_ELIGIBLE",
            [f"Recorded CGPA is {student.cgpa:.2f}; progression to Year {target_year} requires "
             f"a minimum CGPA of {need:.2f} (Clause 12.1, Table 3)."
             + ("" if ok else " The student must use the provisions in Clause 12.4 or 12.5.")],
            [RULE_SOURCE["progression"]],
            {"cgpa": student.cgpa, "required": need, "target_year": target_year},
        )

    def credit_summary(self, student: StudentRecord) -> Verdict:
        cat = self.courses[self.courses.batch.astype(str) == str(student.batch)]
        passed = student.passed()
        earned_rows = cat[cat.course_code.isin(passed)].drop_duplicates("course_code")
        earned = float(earned_rows["credits"].sum())
        if student.credits_earned_override is not None:
            earned = float(student.credits_earned_override)
        by_basket = (earned_rows.groupby("basket")["credits"].sum().to_dict())
        req = self.baskets[self.baskets.batch.astype(str) == str(student.batch)]
        total_req = req[req.basket == "TOTAL"]["min_credits"]
        total_req = float(total_req.iloc[0]) if not total_req.empty else None
        reasons = [f"Credits earned from passed courses on record: {earned:g}"
                   + (f" of {total_req:g} required for the degree." if total_req else ".")]
        unmet = []
        for r in req.itertuples():
            if r.basket == "TOTAL":
                continue
            got = by_basket.get(r.basket, 0.0)
            if got < r.min_credits:
                unmet.append(f"{r.basket}: {got:g}/{r.min_credits:g}")
        if unmet:
            reasons.append("Baskets not yet complete — " + "; ".join(unmet) + ".")
        return Verdict("Credit position", "UNKNOWN" if total_req is None else "ELIGIBLE",
                       reasons, [RULE_SOURCE["credits"]],
                       {"earned": earned, "total_required": total_req, "by_basket": by_basket})

    def failed_course_actions(self, student: StudentRecord) -> Verdict:
        failed = student.failed()
        if not failed:
            return Verdict("Failed-course obligations", "ELIGIBLE",
                           ["No failing grades on record."], [RULE_SOURCE["passing"]])
        reasons, sources = [], []
        for code, grade in failed.items():
            g = str(grade).upper()
            if g == "F":
                reasons.append(
                    f"{code}: grade 'F'. A Make-Up Examination may be permitted in the immediate "
                    f"subsequent semester; if the 'F' stands after the Make-Up, the course must be "
                    f"re-registered whenever it is next offered (Clause 8.13.3 c, e).")
                sources.append(RULE_SOURCE["fail_F"])
            elif g == "FA":
                reasons.append(
                    f"{code}: grade 'FA' (attendance shortage). The course must be re-registered in "
                    f"a subsequent Academic Term until a passing grade is obtained; a Make-Up "
                    f"Examination is not available for 'FA' (Clause 8.13.3 b, g).")
                sources.append(RULE_SOURCE["fail_FA"])
            else:
                reasons.append(f"{code}: grade '{g}' is not a passing grade (Clause 8.11).")
                sources.append(RULE_SOURCE["passing"])
        return Verdict("Failed-course obligations", "NOT_ELIGIBLE", reasons, sources,
                       {"failed": failed})

    # -- the full briefing injected into V4 prompts ------------------------
    def profile_brief(self, student: StudentRecord, mentioned: list[str] | None = None) -> str:
        parts = [
            "=== VERIFIED STUDENT RECORD (computed, not inferred) ===",
            f"Student ID       : {student.student_id}",
            f"Batch / Programme: {student.batch} / {student.programme}",
            f"Current semester : S{student.current_semester} (Academic Year {student.academic_year})",
            f"CGPA             : {student.cgpa if student.cgpa is not None else 'not recorded'}",
            f"Declared minor   : {student.declared_minor or 'none'}",
            f"Passed courses   : {', '.join(sorted(student.passed())) or 'none'}",
            f"Failed courses   : "
            f"{', '.join(f'{c}({g})' for c, g in student.failed().items()) or 'none'}",
        ]
        parts.append("\n=== RULE-ENGINE FINDINGS ===")
        parts.append(self.credit_summary(student).to_prompt_block())
        parts.append(self.check_progression(student).to_prompt_block())
        if student.attendance:
            parts.append(self.check_attendance(student).to_prompt_block())
        if student.failed():
            parts.append(self.failed_course_actions(student).to_prompt_block())
        for code in (mentioned or []):
            v = self.check_prerequisites(student, code)
            parts.append(f"[{v.question}]\n{v.to_prompt_block()}")
            o = self.check_offered(code, student.current_semester + 1, student.batch)
            parts.append(f"[{o.question}]\n{o.to_prompt_block()}")
        return "\n".join(parts)


def load_students(path=None) -> list[StudentRecord]:
    path = path or config.STUDENTS_JSON
    return [StudentRecord(**d) for d in json.load(open(path))]


def save_students(students: list[StudentRecord], path=None) -> None:
    path = path or config.STUDENTS_JSON
    json.dump([asdict(s) for s in students], open(path, "w"), indent=2)
