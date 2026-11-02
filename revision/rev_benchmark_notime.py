"""Table 5 without the timestamp column (revision).

The published Table 5 compared readings of the Hassler corpus on the columns
all five cyber blocks share. Those fourteen columns include timestamp_c, which
is time of day in configuration A and epoch time in configurations B and C, so
on its own it separates the evil twin and false data injection blocks from the
rest. This script repeats every reading with the timestamp removed and keeps
everything else as published (scripts/audit_benchmark.py: same readers, same
grouped and random splits, same models, same seed).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent))
from audit_benchmark import evaluate, read_correct, read_naive  # noqa: E402


def drop_time(d, name):
    keep = [i for i, f in enumerate(d["features"])
            if not f.startswith("timestamp")]
    return dict(d, name=name, X=d["X"][:, keep],
                features=[d["features"][i] for i in keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    path = Path(a.csv)
    naive = read_naive(path)
    c5 = read_correct(path, with_physical=False)
    c5cp = read_correct(path, with_physical=True)
    keep = [i for i, f in enumerate(naive["features"]) if f in set(c5["features"])]
    naive14 = dict(naive, name="naive-3 (14f)", X=naive["X"][:, keep],
                   features=[naive["features"][i] for i in keep])
    readings = [drop_time(naive, "naive-3, no time"),
                drop_time(naive14, "naive-3 (shared), no time"),
                drop_time(c5, "correct-5, no time"),
                drop_time(c5cp, "correct-5 + phys, no time")]
    rows = []
    for d in readings:
        print(d["name"], d["X"].shape, "timestamp columns left:",
              [f for f in d["features"] if f.startswith("timestamp")])
        for sp in ("grouped", "random"):
            rows += evaluate(d, a.seed, sp)
    Path(a.out).write_text(json.dumps({"rows": rows,
                                       "features": {d["name"]: d["features"]
                                                    for d in readings}},
                                      indent=1, default=str))
    for r in rows:
        if r["split"] == "grouped":
            print(f"{r['reading']:<28} {r['model']:<13} acc {r['accuracy']:.4f}")


if __name__ == "__main__":
    main()
