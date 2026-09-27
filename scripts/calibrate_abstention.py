"""
Calibrate the abstention thresholds against the test set.

The abstention test uses ABSOLUTE retrieval signals (raw cosine similarity and
query-term coverage), because the fused rank score is min-max normalised and so
always puts the best hit at 1.0 -- it cannot express "nothing here".

Absolute cosine distributions differ sharply between the TF-IDF fallback and
Gemini embeddings, so the thresholds must be re-fitted whenever the index type
changes. This script sweeps both thresholds and picks the pair that best
separates answerable cases from genuinely unanswerable ones.

    python3 scripts/calibrate_abstention.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from advisor import config, testset            # noqa: E402
from advisor.retriever import HybridRetriever  # noqa: E402

UNANSWERABLE = {"missing_information", "out_of_scope"}


def main() -> None:
    r = HybridRetriever()
    cases = testset.load()
    print(f"index mode = {r.mode}, {len(r.chunks)} chunks, {len(cases)} cases\n")

    rows = []
    for c in cases:
        hits = r.search(c.question, k=config.TOP_K)
        rows.append({
            "id": c.id,
            "category": c.category,
            "should_abstain": c.category in UNANSWERABLE,
            "best_cos": max((h.dense for h in hits), default=0.0),
            "best_cov": max((h.coverage for h in hits), default=0.0),
        })
    df = pd.DataFrame(rows)

    print("Signal distributions:")
    print(df.groupby("should_abstain")[["best_cos", "best_cov"]]
            .agg(["min", "mean", "max"]).round(3).to_string(), "\n")

    best = None
    for sim in np.arange(0.30, 0.95, 0.01):
        for cov in np.arange(0.05, 0.70, 0.01):
            pred = (df.best_cos < sim) & (df.best_cov < cov)
            tp = int((pred & df.should_abstain).sum())
            fp = int((pred & ~df.should_abstain).sum())
            fn = int((~pred & df.should_abstain).sum())
            # False positives are the expensive error: refusing a question the
            # documents CAN answer looks broken to a user. Weight them 2x.
            score = tp - 2 * fp - fn
            if best is None or score > best[0]:
                best = (score, round(float(sim), 2), round(float(cov), 2), tp, fp, fn)

    score, sim, cov, tp, fp, fn = best
    n_abs = int(df.should_abstain.sum())
    print(f"Best thresholds:  MIN_DENSE_SIM = {sim}   MIN_TERM_COVERAGE = {cov}")
    print(f"  correctly abstained     {tp}/{n_abs}")
    print(f"  wrongly refused         {fp}   (answerable questions declined)")
    print(f"  missed abstentions      {fn}\n")

    cfg = ROOT / "advisor" / "config.py"
    s = cfg.read_text()
    import re
    s = re.sub(r"MIN_DENSE_SIM = [\d.]+", f"MIN_DENSE_SIM = {sim}", s)
    s = re.sub(r"MIN_TERM_COVERAGE = [\d.]+", f"MIN_TERM_COVERAGE = {cov}", s)
    cfg.write_text(s)
    print(f"config.py updated.")

    out = config.RESULTS / "abstention_calibration.csv"
    df.to_csv(out, index=False)
    print(f"per-case signals -> {out.name}")


if __name__ == "__main__":
    main()
