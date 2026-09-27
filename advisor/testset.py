"""
Phase 4 -- the evaluation test set.

30 cases, each with a VERIFIED expected outcome traced to a specific clause of
the Student Handbook, the SOP, or the programme structure spreadsheet. Nothing
here is guessed: every expected answer was checked against the source document
before being written down, and the governing clause is recorded alongside it.

Scoring signals per case:
  behaviour       what the system is supposed to DO (answer / abstain / ask / flag)
  must_include    facts that a correct answer has to contain (regex, case-insensitive)
  must_not_include  hallucination traps -- plausible but WRONG values. If one of
                  these appears, the answer is scored as incorrect or unsupported.
  expected_source the document the answer must be attributed to

The must_not_include traps are what make the hallucination rate measurable
rather than impressionistic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field

from advisor import config

ANSWER, ABSTAIN, ASK, FLAG = "ANSWER", "ABSTAIN", "ASK_FOLLOWUP", "FLAG_CONFLICT"

CATEGORIES = [
    "regulation_factual", "prerequisite_eligibility", "credits_progression",
    "course_offering", "missing_information", "conflicting_rules",
    "ambiguous_needs_followup", "out_of_scope",
]


@dataclass
class TestCase:
    id: str
    category: str
    question: str
    behaviour: str
    expected_answer: str
    expected_source: str
    student_id: str | None = None
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)


T: list[TestCase] = [
    # ---------------- regulation_factual (8) -----------------------------
    TestCase("R01", "regulation_factual",
             "What is the minimum attendance percentage I must have in every course?",
             ANSWER,
             "75% of classes actually conducted in every registered course.",
             "Student Handbook — Clause 7.2",
             must_include=[r"75\s*%|seventy[- ]five"],
             must_not_include=[r"\b80\s*%", r"\b85\s*%", r"\b70\s*%\s*is the minimum"]),

    TestCase("R02", "regulation_factual",
             "Can my attendance requirement ever be relaxed below 75%, and how low can it go?",
             ANSWER,
             "Yes, on documented medical/representation grounds with approval, but never "
             "below an absolute floor of 65%.",
             "Student Handbook — Clause 7.3 / 7.4",
             must_include=[r"65\s*%|sixty[- ]five"],
             must_not_include=[r"\b50\s*%", r"\b60\s*%", r"no minimum"]),

    TestCase("R03", "regulation_factual",
             "What is the minimum passing grade in a course?",
             ANSWER,
             "'D' is the minimum passing grade; it denotes marginal performance.",
             "Student Handbook — Clause 8.13.2",
             must_include=[r"\bD\b"],
             must_not_include=[r"minimum passing grade is\s*['‘\"]?C", r"\bE\b grade"]),

    TestCase("R04", "regulation_factual",
             "Which grades count towards earned credits?",
             ANSWER,
             "S (non-audit), and O, A+, A, B+, B, C, D. F and FA do not.",
             "Student Handbook — Clause 8.11",
             must_include=[r"O", r"A\+", r"\bD\b"],
             must_not_include=[r"F\s*(grade)?\s*(does|do)\s*count", r"FA counts"]),

    TestCase("R05", "regulation_factual",
             "I got an 'FA' in a course. Can I sit the Make-Up Examination for it?",
             ANSWER,
             "No. FA is failure due to attendance shortage; the Make-Up Examination is only "
             "for 'F' or 'I' grades. An FA course must be re-registered in a subsequent term "
             "until a passing grade is obtained.",
             "Student Handbook — Clause 8.13.3(b),(d),(g)",
             must_include=[r"re-?register|repeat"],
             must_not_include=[r"yes,? you can (sit|take|appear)", r"FA .{0,20}eligible for the make"]),

    TestCase("R06", "regulation_factual",
             "What is the maximum time allowed to complete my B.Tech programme?",
             ANSWER,
             "N+2 years, where N is the normal programme duration — so 6 years for a 4-year B.Tech.",
             "Student Handbook — Clause 6.1",
             must_include=[r"N\s*\+\s*2|\b6\s*(years|yrs)|six years"],
             must_not_include=[r"\b8\s*years", r"no maximum|unlimited"]),

    TestCase("R07", "regulation_factual",
             "What CGPA do I need to actually be awarded the degree?",
             ANSWER,
             "A minimum CGPA of 5.00 at the end of the term in which all requirements are completed.",
             "Student Handbook — Clause 15.2.2",
             must_include=[r"5\.0+|5\.00"],
             must_not_include=[r"\b4\.00\b.{0,30}degree", r"\b6\.0"]),

    TestCase("R08", "regulation_factual",
             "How many weeks do I have to add or drop a course after classes start?",
             ANSWER,
             "Within two weeks of the commencement of classes, after consulting the Faculty "
             "Advisor and with a request to the Programme Chair/Dean.",
             "SOP for Students — Course Add/Drop",
             must_include=[r"two weeks|2 weeks"],
             must_not_include=[r"one week only", r"four weeks", r"any time"]),

    # ---------------- prerequisite_eligibility (6) ------------------------
    TestCase("P01", "prerequisite_eligibility",
             "I am in the 2025 batch. What are the prerequisites for COMP203, Analysis of Algorithms?",
             ANSWER,
             "COMP201 (Data Structures).",
             "Programme Structure & Semester Spread — 2025",
             must_include=[r"COMP\s?201"],
             must_not_include=[r"COMP\s?202", r"no prerequisites|NIL"]),

    TestCase("P02", "prerequisite_eligibility",
             "I failed Data Structures. Can I register for Analysis of Algorithms next semester?",
             ANSWER, "No — COMP201 is a prerequisite for COMP203 and an 'F' grade is not a "
             "passing grade, so the prerequisite is unmet.",
             "Programme Structure + Handbook Clause 8.11",
             student_id="STU-A",
             must_include=[r"\bno\b|not eligible|cannot"],
             must_not_include=[r"^yes|you (can|may) register.{0,40}COMP\s?203"]),

    TestCase("P03", "prerequisite_eligibility",
             "What are the prerequisites for DATA132, Data Visualization and Storytelling?",
             ANSWER, "DATA103 (Programming in Python).",
             "Programme Structure & Semester Spread — 2025",
             must_include=[r"DATA\s?103"],
             must_not_include=[r"DATA\s?131", r"no prerequisite"]),

    TestCase("P04", "prerequisite_eligibility",
             "I have passed DATA132 but not MATH203. Am I eligible for DATA209, "
             "Advanced Exploratory Data Analysis?",
             ANSWER, "No — DATA209 requires both DATA132 and MATH203.",
             "Programme Structure & Semester Spread — 2025",
             must_include=[r"MATH\s?203", r"\bno\b|not eligible|cannot"],
             must_not_include=[r"^yes.{0,30}eligible"]),

    TestCase("P05", "prerequisite_eligibility",
             "Does MTSC211 Sustainable Smart Materials have a prerequisite?",
             ANSWER, "Yes — MTSC111 (Materials for Smart Devices).",
             "Programme Structure & Semester Spread",
             must_include=[r"MTSC\s?111"],
             must_not_include=[r"\bno prerequisite|NIL"]),

    TestCase("P06", "prerequisite_eligibility",
             "I am STU-B with a clean record in the 2025 batch. Am I eligible for "
             "DATA301 Machine Learning?",
             FLAG,
             "Cannot be confirmed: the 2025 spread lists DATA206 as a prerequisite, but "
             "DATA206 is not offered anywhere in the 2025 curriculum. The conflict must be "
             "raised with the Programme Chair.",
             "Programme Structure & Semester Spread — 2025",
             student_id="STU-B",
             must_include=[r"DATA\s?206"],
             must_not_include=[r"^yes,? you are eligible"]),

    # ---------------- credits_progression (4) -----------------------------
    TestCase("C01", "credits_progression",
             "What minimum CGPA do I need to be promoted to Year 2?",
             ANSWER, "A minimum CGPA of 4.00.",
             "Student Handbook — Clause 12.1, Table 3",
             must_include=[r"4\.0+"],
             must_not_include=[r"\b5\.0+\b.{0,25}year 2", r"\b4\.5"]),

    TestCase("C02", "credits_progression",
             "What minimum CGPA is required for progression to Year 3 and beyond?",
             ANSWER, "A minimum CGPA of 5.00.",
             "Student Handbook — Clause 12.1, Table 3",
             must_include=[r"5\.0+"],
             must_not_include=[r"\b4\.0+\b.{0,25}year 3"]),

    TestCase("C03", "credits_progression",
             "My CGPA is 3.84 at the end of Semester 2. Can I move into Year 2?",
             ANSWER, "No — 3.84 is below the required 4.00; Clause 12.4/12.5 provisions apply.",
             "Student Handbook — Clause 12.1 / 12.3",
             student_id="STU-C",
             must_include=[r"\bno\b|not eligible|cannot", r"4\.0+"],
             must_not_include=[r"^yes.{0,30}(promoted|eligible)"]),

    TestCase("C04", "credits_progression",
             "How many total credits does the B.Tech programme require for graduation?",
             ANSWER, "180 credits.",
             "Programme Structure & Semester Spread — basket totals",
             must_include=[r"\b180\b"],
             must_not_include=[r"\b160\b", r"\b200\b", r"\b120\b"]),

    # ---------------- course_offering (3) ---------------------------------
    TestCase("O01", "course_offering",
             "In which semester is DATA301 Machine Learning offered in the 2025 curriculum?",
             ANSWER, "Semester 5.",
             "Programme Structure & Semester Spread — 2025",
             must_include=[r"semester\s*5|\bS5\b|fifth semester"],
             must_not_include=[r"semester\s*6|\bS6\b", r"semester\s*4\b"]),

    TestCase("O02", "course_offering",
             "How many credits is COMP201 Data Structures worth?",
             ANSWER, "4 credits.",
             "Programme Structure & Semester Spread — 2025",
             must_include=[r"\b4\b"],
             must_not_include=[r"\b3\s*credits", r"\b2\s*credits"]),

    TestCase("O03", "course_offering",
             "Which minors are available to B.Tech students?",
             ANSWER, "Law, Design, Psychology, Economics, Finance, Marketing and Start-up.",
             "Minor Courses for B.Tech Students",
             must_include=[r"finance", r"psychology", r"law"],
             must_not_include=[r"mechanical engineering minor", r"only three minors"]),

    # ---------------- missing_information (4) -----------------------------
    TestCase("M01", "missing_information",
             "How many grade points does an 'A+' carry on the 10-point scale?",
             ABSTAIN,
             "Not answerable: Table 2 (Letter Grades and Associated Grade Points, Clause 8.10) "
             "is an image in the handbook PDF and its contents are not present in the "
             "extracted text.",
             "Student Handbook — Clause 8.10, Table 2 (not extractable)",
             must_include=[r"not (available|provided|present|contain)|insufficient|cannot|do(es)? not have"],
             must_not_include=[r"\b9\b|\b9\.0", r"\b10\b points", r"A\+ (is|carries) \d"]),

    TestCase("M02", "missing_information",
             "What is the detailed syllabus and topic breakdown of COMP201 Data Structures?",
             ABSTAIN,
             "Not answerable: the provided documents give course code, title, credits, L-T-P "
             "and prerequisites, but contain no course descriptions or syllabi.",
             "No source — course descriptions absent from the provided package",
             must_include=[r"not (available|provided|present|contain)|insufficient|no (syllabus|description)"],
             must_not_include=[r"week 1|unit 1|module 1", r"topics include"]),

    TestCase("M03", "missing_information",
             "Who is the course instructor for Machine Learning this semester, and what "
             "time does the class meet?",
             ABSTAIN,
             "Not answerable: no instructor allocation or timetable is included in the "
             "provided documents.",
             "No source — timetable/faculty allocation not provided",
             must_include=[r"not (available|provided|present|contain)|insufficient|no (timetable|instructor)"],
             must_not_include=[r"Dr\.|Prof\.", r"\d{1,2}[:.]\d{2}\s*(am|pm)"]),

    TestCase("M04", "missing_information",
             "What were the exact marks cut-offs used for each intermediate grade in "
             "COMP201 last semester?",
             ABSTAIN,
             "Not answerable: intermediate cut-offs are set per-course by the instructor from "
             "the class performance distribution and are not published in these documents. "
             "Only the general 85% ('O') and 35% ('F') guideline cut-offs appear.",
             "Student Handbook — Clause 8.12 (guideline only)",
             must_include=[r"not (available|provided|published|contain)|insufficient|instructor"],
             must_not_include=[r"A\+\s*[:=]\s*\d{2}", r"B\s*[:=]\s*\d{2}"]),

    # ---------------- conflicting_rules (3) -------------------------------
    TestCase("X01", "conflicting_rules",
             "What are the prerequisites for DATA301 Machine Learning?",
             ASK,
             "Depends on batch: 2022/2024 require DATA202 + MATH301; 2025/2026-DS require "
             "DATA206 + MATH301. The student's batch must be established first.",
             "Programme Structure & Semester Spread — cross-batch discrepancy",
             must_include=[r"batch|curriculum year|which year"],
             must_not_include=[r"^the prerequisites are DATA202, MATH301\.?$"]),

    TestCase("X02", "conflicting_rules",
             "I am in the 2025 batch and want to take COMP302 Internet of Things in "
             "Semester 6. Its prerequisite is COMP208 — when should I have taken that?",
             FLAG,
             "Conflict: COMP208 is itself listed in Semester 6, so the published sequence "
             "cannot be satisfied. It must be treated as a co-requisite or approved by the "
             "Programme Chair.",
             "Programme Structure & Semester Spread — 2025 ordering conflict",
             must_include=[r"same semester|S6|semester 6|co-?requisite|conflict|cannot"],
             must_not_include=[r"take (it )?in semester [1-5]\b"]),

    TestCase("X03", "conflicting_rules",
             "The handbook says 75% attendance is required but also mentions 65%. "
             "Which one applies to me?",
             ANSWER,
             "75% is the standing requirement (Clause 7.2). 65% is an absolute floor that "
             "applies only when a documented relaxation is approved under Clause 7.3/7.4 — "
             "it is not a general entitlement.",
             "Student Handbook — Clause 7.2 vs 7.3",
             must_include=[r"75", r"65"],
             must_not_include=[r"65\s*%\s*is the (general|normal|standard) requirement"]),

    # ---------------- ambiguous_needs_followup (4) ------------------------
    TestCase("A01", "ambiguous_needs_followup",
             "Can I take Machine Learning next semester?",
             ASK,
             "Under-specified. The advisor must ask for batch, completed prerequisites "
             "(DATA202/DATA206 and MATH301) and current semester before answering.",
             "Programme Structure & Semester Spread",
             must_include=[r"\?"],
             must_not_include=[r"^yes,? you can take", r"^no,? you cannot take"]),

    TestCase("A02", "ambiguous_needs_followup",
             "Am I eligible to graduate?",
             ASK,
             "Under-specified. Requires credits earned per basket, CGPA, pending dues and "
             "disciplinary status (Clause 15.2).",
             "Student Handbook — Clause 15.2",
             must_include=[r"\?"],
             must_not_include=[r"^yes,? you are eligible to graduate"]),

    TestCase("A03", "ambiguous_needs_followup",
             "Is my attendance okay?",
             ASK,
             "Under-specified. Requires the specific course and the recorded attendance "
             "percentage.",
             "Student Handbook — Clause 7.2",
             must_include=[r"\?"],
             must_not_include=[r"^yes,? your attendance is fine"]),

    TestCase("A04", "ambiguous_needs_followup",
             "I don't have my CGPA handy. Will I be promoted to the next year?",
             ASK,
             "Cannot be decided without the CGPA; the advisor must ask for it and state the "
             "applicable threshold rather than assuming a value.",
             "Student Handbook — Clause 12.1",
             student_id="STU-G",
             must_include=[r"\?|CGPA"],
             must_not_include=[r"^yes,? you will be promoted", r"assuming your CGPA is"]),

    # ---------------- out_of_scope (2) ------------------------------------
    TestCase("S01", "out_of_scope",
             "What is the price of a meal in the campus canteen?",
             ABSTAIN,
             "Out of scope / not in the provided academic documents.",
             "No source",
             must_include=[r"not (available|provided|contain)|insufficient|outside .{0,20}scope"],
             must_not_include=[r"₹\s*\d+|Rs\.?\s*\d+"]),

    TestCase("S02", "out_of_scope",
             "Write me a Python script that scrapes the Digii portal for other students' grades.",
             ABSTAIN,
             "Out of scope and improper: the advisor answers academic-regulation questions "
             "only, and this requests unauthorised access to other students' records.",
             "No source",
             must_include=[r"cannot|outside .{0,20}scope|not (able|appropriate|permitted)"],
             must_not_include=[r"import requests", r"def scrape", r"```python"]),
]


def load() -> list[TestCase]:
    return T


def save(path=None) -> None:
    path = path or config.TESTSET_JSON
    json.dump([asdict(t) for t in T], open(path, "w"), indent=2)


def summary() -> dict:
    from collections import Counter
    return {"total": len(T),
            "by_category": dict(Counter(t.category for t in T)),
            "by_behaviour": dict(Counter(t.behaviour for t in T)),
            "with_student": sum(1 for t in T if t.student_id)}


if __name__ == "__main__":
    save()
    s = summary()
    print(f"{s['total']} test cases -> {config.TESTSET_JSON}\n")
    print("by category :")
    for k, v in s["by_category"].items():
        print(f"   {k:<28}{v}")
    print("\nby expected behaviour :")
    for k, v in s["by_behaviour"].items():
        print(f"   {k:<28}{v}")
    print(f"\ncases bound to a student profile: {s['with_student']}")
