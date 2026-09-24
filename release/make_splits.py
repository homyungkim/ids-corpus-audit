#!/usr/bin/env python3
"""Write reference splits so results on this corpus are comparable.

Three policies are emitted for every reading, and each is a list of row indices
rather than a rule, so nobody has to reimplement the grouping and get it subtly
different.

  grouped-3000   whole stretches of ~3000 consecutive rows held out. This is
                 the policy to report by default: it is the only one of the
                 three under which a test row's neighbours are not in training.
  grouped-500    a middle setting, for comparison
  random-row     what most published results on this corpus used. Emitted so
                 that the gap can be quantified, not so that it can be used.

Usage:
    python release/make_splits.py --release release/data/v1.0.0 --out release/splits
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uav_cpids import load  # noqa: E402

READINGS = [
    {"name": "cyber-5", "kw": {"view": "cyber"}},
    {"name": "cyber-3-captureA", "kw": {"view": "cyber", "capture": "A"}},
    {"name": "both-3-captureA-stretch",
     "kw": {"view": "both", "capture": "A", "pair": "stretch"}},
    {"name": "both-3-captureA-truncate",
     "kw": {"view": "both", "capture": "A", "pair": "truncate"}},
]
SEEDS = (0, 1, 2, 3, 4)
TEST_SIZE = 0.3


def test_groups(g, test_size, seed) -> list[int]:
    rng = np.random.default_rng(seed)
    groups = np.unique(g)
    rng.shuffle(groups)
    return sorted(int(x) for x in
                  groups[:max(1, int(round(len(groups) * test_size)))])


def random_row_digest(n_rows, test_size, seed) -> str:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_rows)
    te = np.sort(idx[:int(round(n_rows * test_size))])
    return hashlib.sha256(te.tobytes()).hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release/data/v1.0.0")
    ap.add_argument("--out", default="release/splits")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    index = {"test_size": TEST_SIZE, "seeds": list(SEEDS),
             "policies": {
                 "grouped-3000": "whole stretches of ~3000 consecutive rows "
                                 "held out; report this one",
                 "grouped-500": "middle setting, for comparison",
                 "random-row": "row-level shuffle; what most published results "
                               "used. A test row's neighbours are in training.",
             },
             "readings": []}

    for r in READINGS:
        for block in (3000, 500):
            d = load(a.release, block_rows=block, **r["kw"])
            for seed in SEEDS:
                (out / f"{r['name']}__grouped-{block}__seed{seed}.json"
                 ).write_text(json.dumps({
                     "policy": f"grouped-{block}", "reading": r["name"],
                     "load_kwargs": r["kw"], "block_rows": block,
                     "test_size": TEST_SIZE, "seed": seed,
                     "test_groups": test_groups(d.group, TEST_SIZE, seed),
                 }, indent=1) + "\n")
        d = load(a.release, block_rows=3000, **r["kw"])
        for seed in SEEDS:
            (out / f"{r['name']}__random-row__seed{seed}.json"
             ).write_text(json.dumps({
                 "policy": "random-row", "reading": r["name"],
                 "load_kwargs": r["kw"], "block_rows": None,
                 "test_size": TEST_SIZE, "seed": seed, "n_rows": len(d),
                 "test_sha256_16": random_row_digest(len(d), TEST_SIZE, seed),
                 "note": "reconstructed by uav_cpids.apply_split; present for "
                         "comparison against the grouped policies, not for "
                         "building on",
             }, indent=1) + "\n")

        index["readings"].append({
            "name": r["name"], "load_kwargs": r["kw"], "rows": len(d),
            "classes": d.counts(),
            "n_groups_3000": int(len(np.unique(d.group))),
        })
        print(f"  {r['name']:<28} {len(d):>7,} rows  "
              f"{len(np.unique(d.group)):>3} groups at 3000")

    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"\nwrote {out} ({len(list(out.glob('*.json'))) - 1} split files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
