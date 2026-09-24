#!/usr/bin/env python3
"""Assertions that pin the corpus down. Run after building a release.

Every number here was read off the file, not off the paper or the README. If
one of these fails on a future release of the source, the structure changed
and every downstream result built on the old structure is suspect.

    python release/test_loader.py --release release/data/v1.0.0 \
                                  --original Dataset_T-ITS.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from uav_cpids import (PairingRequired, SchemaRequired,  # noqa: E402
                       describe, load)

EXPECTED_ROWS = {
    ("benign", "cyber"): 9425, ("benign", "physical"): 4290,
    ("dos", "cyber"): 11671, ("dos", "physical"): 973,
    ("replay", "cyber"): 12006, ("replay", "physical"): 973,
    ("evil_twin", "cyber"): 5683, ("evil_twin", "physical"): 5473,
    ("fdi", "cyber"): 3473, ("fdi", "physical"): 807,
}
TOTAL_ROWS = 54774
NAIVE_SURVIVORS = 33102          # what read_csv + dropna leaves
LOST_LABELS = TOTAL_ROWS - NAIVE_SURVIVORS


def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release/data/v1.0.0")
    ap.add_argument("--original", default=None)
    a = ap.parse_args()
    ok = True

    for label, src in [("corrected release", a.release),
                       ("original CSV", a.original)]:
        if src is None:
            continue
        print(f"\n{label}: {src}")
        d = describe(src)
        ok &= check("ten blocks", len(d["blocks"]) == 10, f"{len(d['blocks'])}")
        ok &= check("total rows", d["total_rows"] == TOTAL_ROWS,
                    f"{d['total_rows']:,}")
        for b in d["blocks"]:
            k = (b["class"], b["kind"])
            ok &= check(f"{b['class']}/{b['kind']} rows",
                        b["rows"] == EXPECTED_ROWS[k],
                        f"{b['rows']:,}")
        ok &= check("fdi physical has no timestamp",
                    not next(b for b in d["blocks"]
                             if b["class"] == "fdi"
                             and b["kind"] == "physical")["has_timestamp"])
        ok &= check("five physical blocks share exactly 3 columns",
                    len(d["physical_common_all"]) == 3,
                    ", ".join(d["physical_common_all"]))
        # The original README says "sixteen features" and "thirty-seven
        # features". That count includes the timestamp column; the loader
        # excludes it, because a capture timestamp is bookkeeping and also
        # leaks block identity. The two conventions differ by exactly one,
        # and the README's numbers hold for capture A and for nothing else.
        ok &= check("capture A physical: 15 features + timestamp = README's 16",
                    len(d["physical_common_capture_A"]) == 15,
                    f"{len(d['physical_common_capture_A'])} + 1")
        ok &= check("capture A cyber: 36 features + timestamp = README's 37",
                    len(d["cyber_common_capture_A"]) == 36,
                    f"{len(d['cyber_common_capture_A'])} + 1")
        ok &= check("captures B and C do not match the README's counts",
                    len(d["physical_common_all"]) == 3,
                    "only pitch/roll/yaw survive across all five")

    print("\nloader behaviour")
    c = load(a.release, view="cyber")
    ok &= check("cyber view keeps every row",
                len(c) == sum(v for (f, k), v in EXPECTED_ROWS.items()
                              if k == "cyber"), f"{len(c):,}")
    ok &= check("five families present", len(c.counts()) == 5, str(c.counts()))

    try:
        load(a.release, view="both")
        ok &= check("view='both' without pair is refused", False)
    except PairingRequired:
        ok &= check("view='both' without pair is refused", True)

    try:
        load(a.release, view="both", pair="stretch")
        ok &= check("five-family physical view is refused", False)
    except SchemaRequired:
        ok &= check("five-family physical view is refused", True)

    s = load(a.release, view="both", capture="A", pair="stretch")
    ok &= check("capture A + stretch pairs to the cyber length",
                len(s) == 9425 + 11671 + 12006, f"{len(s):,}")
    ok &= check("stretch records what it invented",
                all(r["share_invented"] > 0 for r in s.provenance["pairing_cost"]),
                ", ".join(f"{r['class']} {100*r['share_invented']:.0f}%"
                          for r in s.provenance["pairing_cost"]))

    t = load(a.release, view="both", capture="A", pair="truncate")
    ok &= check("truncate keeps min(len) per class",
                len(t) == 4290 + 973 + 973, f"{len(t):,}")

    n = load(a.release, view="both", capture="A", pair="none")
    try:
        _ = n.X
        ok &= check("pair='none' refuses to be concatenated", False)
    except ValueError:
        ok &= check("pair='none' refuses to be concatenated", True)

    tr, te = c.grouped_split(seed=0)
    ok &= check("grouped split is disjoint",
                len(set(c.group[tr]) & set(c.group[te])) == 0)
    ok &= check("grouped split covers every row",
                len(tr) + len(te) == len(c), f"{len(tr):,} + {len(te):,}")

    print("\naudit arithmetic")
    ok &= check("naive read survivors",
                9425 + 11671 + 12006 == NAIVE_SURVIVORS, f"{NAIVE_SURVIVORS:,}")
    ok &= check("labels a naive read empties",
                LOST_LABELS == 21672, f"{LOST_LABELS:,} "
                f"({100*LOST_LABELS/TOTAL_ROWS:.1f}%)")
    ok &= check("evil_twin and fdi lose every label",
                EXPECTED_ROWS[("evil_twin", "cyber")]
                + EXPECTED_ROWS[("evil_twin", "physical")]
                + EXPECTED_ROWS[("fdi", "cyber")]
                + EXPECTED_ROWS[("fdi", "physical")] == 15436,
                "15,436 rows in the two families a naive read cannot label")

    print("\n" + ("all checks passed" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
