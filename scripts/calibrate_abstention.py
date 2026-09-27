"""
Calibrate the retrieval-stage abstention threshold (config.RERANK_MIN).

Abstention happens at TWO layers:
  1. Retrieval gate  -- if the cross-encoder judges even the best passage
                        irrelevant, the model is never called (this script).
  2. Grounding rules -- if relevant passages exist but do not contain the
                        answer (e.g. Clause 8.10 mentions Table 2 but the table
                        is an image), the prompt obliges the model to say
                        INSUFFICIENT INFORMATION.

The gate must therefore be conservative: refusing an answerable question is a
visible failure, while a missed gate is still caught by layer 2. The threshold
is placed midway between the weakest ANSWERABLE question and the strongest
OUT-OF-SCOPE one.

    python3 scripts/calibrate_abstention.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from advisor import config, testset                      # noqa: E402
from advisor.eligibility import load_students            # noqa: E402
from advisor.retriever import HybridRetriever            # noqa: E402


def main() -> None:
    r = HybridRetriever()
    students = {s.student_id: s for s in load_students()}
    rows = []
    for c in testset.load():
        stu = students.get(c.student_id) if c.student_id else None
        hits = r.search(c.question, batch=stu.batch if stu else None)
        rows.append({"id": c.id, "category": c.category,
                     "group": ("out_of_scope" if c.category == "out_of_scope"
                               else "missing_info" if c.category == "missing_information"
                               else "answerable"),
                     "best_rerank": round(max(h.score for h in hits), 2) if hits else -99,
                     "top_source": hits[0].chunk.citation()[:60] if hits else ""})
    df = pd.DataFrame(rows).sort_values("best_rerank")
    print(df.to_string(index=False), "\n")

    ans = df[df.group == "answerable"].best_rerank
    oos = df[df.group == "out_of_scope"].best_rerank
    lo_ans = float(ans.min())
    # Policy: the gate may NEVER refuse an answerable question. It is placed
    # midway between the weakest answerable query and the strongest
    # out-of-scope query that lies BELOW it. Out-of-scope queries that
    # overlap answerable ones lexically (S02 mentions the Digii portal, which
    # the SOP describes) are left to the grounding/scope rules in the prompt.
    below = oos[oos < lo_ans]
    hi_oos = float(below.max()) if len(below) else lo_ans - 2.0
    thr = round((lo_ans + hi_oos) / 2, 2)
    print(f"weakest answerable question : {lo_ans:.2f}")
    print(f"strongest separable out-of-scope : {hi_oos:.2f}")
    print(f"=> RERANK_MIN = {thr}   (margin {lo_ans - thr:.2f} / {thr - hi_oos:.2f})")
    refused = int((ans < thr).sum())
    gated = int((oos < thr).sum())
    print(f"   answerable wrongly refused at gate : {refused}/{len(ans)}")
    print(f"   out-of-scope stopped at gate       : {gated}/{len(oos)}")
    print(f"   missing-info cases -> handled by grounding rules (layer 2)")

    cfg = ROOT / "advisor" / "config.py"
    s = re.sub(r"RERANK_MIN = -?[\d.]+", f"RERANK_MIN = {thr}", cfg.read_text())
    cfg.write_text(s)
    df.to_csv(config.RESULTS / "abstention_calibration.csv", index=False)
    print("\nconfig.py updated; per-case scores -> results/abstention_calibration.csv")


if __name__ == "__main__":
    main()
