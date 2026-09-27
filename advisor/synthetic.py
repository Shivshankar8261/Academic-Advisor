"""
Phase 2 -- synthetic, fully anonymised student profiles.

No real student data is used anywhere. Identifiers are opaque tokens
(STU-A ... STU-J), there are no names, no e-mail addresses, no phone numbers,
no dates of birth and no enrolment numbers. Every course code and credit value,
however, is taken from the real programme structure, so the eligibility
arithmetic these profiles exercise is genuine rather than made up.

Each profile is engineered to hit one of the edge cases the assignment lists:
missing information, ambiguity, insufficient prerequisites, insufficient
credits, unavailable courses, and conflicting academic rules.
"""
from __future__ import annotations

import ast

import pandas as pd

from advisor import config
from advisor.eligibility import StudentRecord, save_students


def _catalogue() -> pd.DataFrame:
    df = pd.read_csv(config.COURSES_CSV)
    df["prereqs"] = df["prereqs"].apply(
        lambda v: ast.literal_eval(v) if isinstance(v, str) and v.startswith("[") else [])
    return df


def _transcript(cat: pd.DataFrame, batch: str, upto_sem: int,
                default: str = "B", overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Grades for every course the batch offers up to and including `upto_sem`."""
    sub = cat[(cat.batch.astype(str) == batch) & (cat.sem_num <= upto_sem)]
    out = {r.course_code: default for r in sub.itertuples()}
    out.update(overrides or {})
    return out


def build_profiles() -> list[StudentRecord]:
    cat = _catalogue()
    P: list[StudentRecord] = []

    # 1. The profile named in the assignment brief: passed Linear Algebra and
    #    Programming in Python, failed Data Structures.
    P.append(StudentRecord(
        "STU-A", "2025", "B.Tech Computer Science (Data Science)", 5, 6.42,
        completed=_transcript(cat, "2025", 4, "B", {
            "MATH209": "A",      # Linear Algebra  -> passed
            "DATA103": "A+",     # Programming in Python -> passed
            "COMP201": "F",      # Data Structures -> FAILED
        }),
        attendance={"COMP201": 71.0, "COMP203": 88.0},
        notes="Edge case: failed a course that is a prerequisite for COMP203."))

    # 2. Strong student -- exercises the DATA301 dangling-prerequisite conflict.
    P.append(StudentRecord(
        "STU-B", "2025", "B.Tech Computer Science (Data Science)", 5, 8.71,
        completed=_transcript(cat, "2025", 4, "A"),
        attendance={c: 92.0 for c in ("COMP301", "MATH301")},
        notes="Edge case: clean record, but DATA301 requires DATA206 which the "
              "2025 spread never offers."))

    # 3. Below the Year-2 progression threshold (Clause 12.1 needs CGPA 4.00).
    P.append(StudentRecord(
        "STU-C", "2025", "B.Tech Computer Science (Data Science)", 2, 3.84,
        completed=_transcript(cat, "2025", 2, "C", {"COMP201": "F", "MATH203": "F"}),
        attendance={"COMP201": 69.0, "MATH203": 74.0},
        notes="Edge case: insufficient CGPA for progression to Year 2."))

    # 4. Below the Year-3 progression threshold (needs 5.00), only just.
    P.append(StudentRecord(
        "STU-D", "2025", "B.Tech Computer Science (Data Science)", 4, 4.93,
        completed=_transcript(cat, "2025", 4, "D", {"COMP205": "F"}),
        attendance={"COMP205": 76.0},
        notes="Edge case: borderline CGPA 4.93 against a 5.00 requirement."))

    # 5. Attendance below the absolute 65% floor -> cannot be relaxed at all.
    P.append(StudentRecord(
        "STU-E", "2025", "B.Tech Computer Science (Data Science)", 3, 5.90,
        completed=_transcript(cat, "2025", 2, "B", {"COMP201": "FA"}),
        attendance={"COMP201": 58.0, "COMP209": 80.0, "MATH203": 63.0},
        notes="Edge case: 'FA' grade plus attendance under the hard 65% floor."))

    # 6. Attendance in the 65-75% band -> conditional, needs approval.
    P.append(StudentRecord(
        "STU-F", "2025", "B.Tech Computer Science (Data Science)", 3, 6.80,
        completed=_transcript(cat, "2025", 2, "B"),
        attendance={"COMP203": 70.0, "MATH209": 72.0, "COMP202": 91.0},
        notes="Edge case: attendance shortfall that is relaxable only with approval."))

    # 7. Missing information -- no CGPA on record, so progression is unanswerable.
    P.append(StudentRecord(
        "STU-G", "2025", "B.Tech Computer Science (Data Science)", 4, None,
        completed=_transcript(cat, "2025", 3, "B"),
        attendance={},
        notes="Edge case: CGPA and attendance absent; advisor must ask, not assume."))

    # 8. Older batch -- same course code, different prerequisites (CROSS_BATCH).
    P.append(StudentRecord(
        "STU-H", "2022", "B.Tech Computer Science (Data Science)", 5, 7.10,
        completed=_transcript(cat, "2022", 4, "B", {"DATA202": "A", "MATH301": "B+"}),
        attendance={"DATA301": 85.0},
        notes="Edge case: 2022 curriculum, where DATA301 requires DATA202 + MATH301."))

    # 9. Declared minor -- exercises the Minor/Open basket requirement.
    P.append(StudentRecord(
        "STU-I", "2025", "B.Tech Computer Science (Data Science)", 5, 7.55,
        completed=_transcript(cat, "2025", 4, "B"),
        attendance={}, declared_minor="Finance",
        notes="Edge case: minor credits sit outside the parsed core spread."))

    # 10. Ambiguous -- very few credits, unclear which curriculum applies.
    P.append(StudentRecord(
        "STU-J", "2025", "B.Tech Computer Science (Data Science)", 5, 5.05,
        completed=_transcript(cat, "2025", 2, "C", {"COMP201": "F", "COMP209": "FA"}),
        attendance={"COMP209": 61.0},
        credits_earned_override=72.0,
        notes="Edge case: 72 credits earned but in S5; carries both an 'F' and an 'FA'."))
    return P


def main() -> list[StudentRecord]:
    profiles = build_profiles()
    save_students(profiles)
    from advisor.eligibility import EligibilityEngine
    eng = EligibilityEngine()
    print(f"{len(profiles)} synthetic profiles -> {config.STUDENTS_JSON}\n")
    print(f"{'ID':<7}{'batch':<9}{'sem':<5}{'CGPA':<7}{'passed':<8}{'failed':<8}{'credits':<9}notes")
    for p in profiles:
        cs = eng.credit_summary(p)
        print(f"{p.student_id:<7}{p.batch:<9}S{p.current_semester:<4}"
              f"{str(p.cgpa or '-'):<7}{len(p.passed()):<8}{len(p.failed()):<8}"
              f"{cs.facts['earned']:<9.0f}{p.notes[:52]}")
    return profiles


if __name__ == "__main__":
    main()
