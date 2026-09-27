"""
Run the full Phase 4 evaluation from the command line.

    export GOOGLE_API_KEY=...
    python3 scripts/run_evaluation.py            # all four variants, 34 cases
    python3 scripts/run_evaluation.py --quick    # V1 and V4 only
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from advisor import config, evaluate, prompts, testset      # noqa: E402
from advisor.advisor import build_all_variants              # noqa: E402
from advisor.eligibility import load_students               # noqa: E402


def charts(table: pd.DataFrame, per_cat: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = [prompts.VARIANT_LABELS[v] for v in prompts.VARIANTS if
             prompts.VARIANT_LABELS[v] in table.index]
    t = table.reindex(order)
    short = [l.split("·")[0].strip() for l in order]

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for a, (col, title, colour) in zip(ax.ravel(), [
        ("accuracy_%", "Accuracy (%)", "#4a6fa5"),
        ("hallucination_rate_%", "Hallucination rate (%) — lower is better", "#c0504d"),
        ("missing_info_handled_%", "Missing information handled (%)", "#548235"),
        ("source_correct_%", "Source / evidence correctness (%)", "#8064a2"),
    ]):
        vals = pd.to_numeric(t[col], errors="coerce").fillna(0).values
        bars = a.bar(short, vals, color=colour)
        a.set_title(title, fontsize=11)
        a.set_ylim(0, 105)
        a.grid(axis="y", alpha=.3)
        a.bar_label(bars, fmt="%.0f", padding=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(config.RESULTS / "phase5_progression.png", dpi=160)
    plt.close()

    fig, a = plt.subplots(figsize=(11, 5))
    per_cat.reindex(columns=order).plot(kind="bar", ax=a, width=.8, colormap="viridis")
    a.set_ylabel("% correct")
    a.set_ylim(0, 105)
    a.set_title("Correctness by question category")
    a.grid(axis="y", alpha=.3)
    a.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(config.RESULTS / "phase5_by_category.png", dpi=160)
    plt.close()
    print("charts -> results/phase5_progression.png, results/phase5_by_category.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="V1 and V4 only")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="extra seconds between calls (the LLM router already throttles)")
    ap.add_argument("--max-wait", type=float, default=65.0,
                    help="seconds to wait for Groq's minute window before falling back")
    args = ap.parse_args()

    if not (config._secret("GROQ_API_KEY") or config._secret("GOOGLE_API_KEY")):
        sys.exit("No GROQ_API_KEY / GOOGLE_API_KEY found in environment or .env")

    if not config.CHUNKS_JSONL.exists():
        from advisor.ingest import build_all
        build_all()
    if not config.STUDENTS_JSON.exists():
        from advisor.synthetic import main as gen
        gen()

    cases = testset.load()
    testset.save()
    students = {s.student_id: s for s in load_students()}
    # Evaluation WAITS for Groq's per-minute window instead of falling back,
    # so that (quota permitting) one model answers every case.
    variants = build_all_variants(max_wait=args.max_wait)
    if args.quick:
        variants = {k: v for k, v in variants.items() if k in ("V1_basic", "V4_rag_student")}

    print(f"{len(cases)} cases x {len(variants)} variants "
          f"= {len(cases)*len(variants)} model calls\n")

    results = {}
    for name, adv in variants.items():
        print(f"===== {prompts.VARIANT_LABELS[name]} =====")
        results[name] = evaluate.run_variant(adv, cases, students, sleep=args.sleep)

    paths = evaluate.save_results(results)
    table = evaluate.comparison_table(results)
    per_cat = evaluate.per_category(results)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 78)
    print("THE EIGHT METRICS")
    print("=" * 78)
    print(table.T.to_string())
    print("\n" + "=" * 78)
    print("CORRECTNESS BY CATEGORY (%)")
    print("=" * 78)
    print(per_cat.to_string())

    charts(table, per_cat)
    raw = pd.concat([pd.DataFrame([r.__dict__ for r in res]) for res in results.values()])
    print("\nanswers by provider/model:")
    print(raw.groupby(["variant", "model"]).size().to_string())
    print("\nsaved:", ", ".join(str(p.name) for p in paths.values()))
    print("\nNext:  python3 scripts/build_report.py")


if __name__ == "__main__":
    main()
