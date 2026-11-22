"""Command-line evaluation and aggregate report for TalentFlow AI.

Examples:
    python scripts/evaluate.py                          # synthetic demo (1,800 records, seed 42)
    python scripts/evaluate.py --csv my_anonymised.csv  # your own file (same validation as the app)
    python scripts/evaluate.py --out reports/           # also write aggregate CSV/JSON reports
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from talentflow.core import (BANDS, LABELS, MIN_GROUP, band_rates, evaluate, evaluation_summary, generate,  # noqa: E402
                             group_rates, prepare, yearly)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", help="anonymised workforce CSV (default: synthetic demo)")
    ap.add_argument("--n", type=int, default=1800, help="synthetic records (default 1800)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-group", type=int, default=MIN_GROUP, help="smallest group reported (>= 10)")
    ap.add_argument("--out", help="folder for aggregate reports")
    a = ap.parse_args()
    min_n = max(a.min_group, MIN_GROUP)

    if a.csv:
        d, rep = prepare(pd.read_csv(a.csv, low_memory=False))
        print(rep.as_frame().to_string(index=False))
    else:
        d = generate(a.n, a.seed)
    ev = evaluate(d)
    m = ev.metrics
    print(f"\nRecords {len(d):,} · turnover {d.left.mean():.1%} · train {m['train_rows']:,} / holdout {m['test_rows']:,}")
    print(f"Holdout ROC-AUC {m['roc_auc']:.3f} · average precision {m['average_precision']:.3f} · Brier {m['brier']:.3f}")
    print(f"Threshold {m['threshold']:.2f} (chosen on training folds): precision {m['precision']:.3f} · "
          f"recall {m['recall']:.3f} · F1 {m['f1']:.3f}")
    print("\nModel comparison (same holdout):")
    print(ev.comparison.round(3).to_string(index=False))
    if ev.temporal:
        t = ev.temporal
        print(f"\nTime-ordered check {t['train_years']} -> {t['test_year']}: ROC-AUC {t['roc_auc']:.3f}, "
              f"mean score {t['mean_score']:.1%} vs observed {t['observed_rate']:.1%}")
    print("\nPermutation importance (holdout AUC drop):")
    print(ev.importance[["label", "auc_drop", "std"]].round(4).to_string(index=False))

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        dep, hid = group_rates(d, "department", min_n)
        dep.round(4).to_csv(out / "department_summary.csv", index=False)
        bands = pd.concat([band_rates(d, c, min_n)[0].assign(factor=LABELS[c]) for c in BANDS], ignore_index=True)
        bands.round(4).to_csv(out / "factor_band_summary.csv", index=False)
        if "year" in d.columns:
            yearly(d, min_n)[0].round(4).to_csv(out / "yearly_turnover.csv", index=False)
        (out / "evaluation.json").write_text(json.dumps(evaluation_summary(ev), indent=2, default=str))
        print(f"\nAggregate reports written to {out}/ ({hid} small departments suppressed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
