#!/usr/bin/env python3
"""How stable is the capture-fingerprint index?

The index of (1) is reported per corpus as a single number, and a single
number invites the reading that a corpus at 0.914 differs from one at 0.917.
Two things vary between runs: which records a subsampled corpus draws, and
which of them land in the test half. This script varies both and reports the
spread, so the paper can state the index to a resolution it actually has.

    python audit/fingerprint_variance.py --hassler /path/Dataset_T-ITS.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters                              # noqa: E402
from adapters import ADAPTERS                # noqa: E402
from checklist import C4_capture_fingerprint  # noqa: E402
from make_reference_splits import load_digest   # noqa: E402

DRAWS = 8
SPLITS = 3
SPLITS_IF_NO_SUBSAMPLE = 24


def index_of(result) -> float:
    """C4 reports the index first in its value string."""
    return float(str(result.value).split()[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ADAPTERS:
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--out", default="runs/audit/fingerprint_variance.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    real = adapters._subsample
    out = []
    for key, fn in ADAPTERS.items():
        path = getattr(a, key)
        if not path:
            continue
        # A corpus smaller than the record cap is not subsampled at all, so
        # varying the draw would reload identical records and report a spread
        # over repeated values. Detect that and spend the budget on splits
        # instead, which is the only dimension that varies there.
        adapters._subsample = real
        base = fn(path)
        adapters._subsample = lambda n, cap, k: real(n, cap, f"{k}/draw1")
        subsampled = load_digest(fn(path)) != load_digest(base)
        adapters._subsample = real

        draws = DRAWS if subsampled else 1
        splits = SPLITS if subsampled else SPLITS_IF_NO_SUBSAMPLE
        if not subsampled:
            logging.info("  %s is not subsampled; varying the split only", key)

        vals = []
        for d in range(draws):
            # vary the draw by varying the key the seed is derived from; the
            # adapter is left untouched so the sampling stays the real one
            adapters._subsample = (
                lambda n, cap, k, _d=d: real(n, cap, f"{k}/draw{_d}"))
            r = fn(path)
            for s in range(splits):
                vals.append(index_of(C4_capture_fingerprint(r, seed=s)))
            logging.info("  %s draw %d: %s", key, d,
                         " ".join(f"{v:.4f}" for v in vals[-splits:][:8]))
        adapters._subsample = real
        v = np.array(vals)
        rec = {"corpus": key, "draws": draws, "splits_per_draw": splits,
               "subsampled": bool(subsampled),
               "n": len(v), "mean": float(v.mean()), "sd": float(v.std()),
               "min": float(v.min()), "max": float(v.max()),
               "values": [round(x, 4) for x in v.tolist()]}
        out.append(rec)
        logging.info("%s: %.4f +- %.4f  (range %.4f..%.4f, n=%d)",
                     key, rec["mean"], rec["sd"], rec["min"], rec["max"],
                     rec["n"])

    if not out:
        print("nothing measured")
        return 1
    print("\n" + "=" * 72)
    print(f"{'corpus':<16}{'mean':>9}{'sd':>9}{'min':>9}{'max':>9}"
          f"{'report as':>20}")
    print("-" * 72)
    for r in out:
        # round to the first decimal place the spread supports
        dp = 2 if r["sd"] >= 0.005 else 3
        print(f"{r['corpus']:<16}{r['mean']:>9.4f}{r['sd']:>9.4f}"
              f"{r['min']:>9.4f}{r['max']:>9.4f}"
              f"{r['mean']:>14.{dp}f} +- {r['sd']:.{dp}f}")
    print("=" * 72)
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
