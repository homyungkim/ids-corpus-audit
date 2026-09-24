#!/usr/bin/env python3
"""Structural audit of the HCRL MAVLink message-ID sequence corpus.

    github.com/0pt3ryx/GUIDE  (data/raw/*.npy)
    J. D. Yoo, H. Kim, H. K. Kim, "GUIDE: GAN-Based UAV IDS Enhancement,"
    Computers & Security 147:104073, 2024, doi:10.1016/j.cose.2024.104073

Each file is a flat 1-D sequence of MAVLink message IDs; the class lives in
the file name, not in the data. Two properties of the released pipeline follow
from that and are measured here.

1. Windows overlap by 127 of 128 elements, and are then split at random.
   `data_preparation.py` builds windows with
   `TimeseriesGenerator(sequence, ..., length=128)`, which emits every stride-1
   window, then calls
   `train_test_split(..., test_size=0.2, random_state=777, shuffle=True)`.
   A test window and the window one message later differ in a single element
   and land on opposite sides of the split. `_filter_unique` removes exact
   duplicates from the training half only; near-duplicates across the split
   are untouched.

2. The label is the capture, so "attack" traffic is mostly normal traffic.
   A MAVLink capture taken while a ping flood is running still carries the
   ordinary heartbeat and telemetry stream. If a model can tell two captures
   of the SAME class apart as easily as it tells normal from attack, then what
   it has learned is which capture a window came from.

The old dataset ships two captures per class (normal / normal2, ping / ping2,
...), which makes test 2 possible without any extra data.

Usage:
    python scripts/audit_mavlink.py --raw /path/to/GUIDE/data/raw
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("mavlink")

WINDOW = 128            # config_parameter.json: sequence_length_list = [128]
TEST_SIZE = 0.2         # data_preparation.py
PAIRS = [("normal", "normal2"), ("heartbeat", "heartbeat2"),
         ("ping", "ping2"), ("request", "request2")]


def windows(seq: np.ndarray, n_take: int, w: int = WINDOW) -> np.ndarray:
    """Every stride-1 window of the first n_take elements, as the code does."""
    s = seq[:n_take].astype(np.int16)
    n = len(s) - w + 1
    return np.lib.stride_tricks.sliding_window_view(s, w)[:n]


def fit_predict(Xtr, ytr, Xte, yte, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    m = RandomForestClassifier(n_estimators=120, n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr)
    return float(accuracy_score(yte, m.predict(Xte)))


def random_split(X, y, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n = int(round(len(y) * TEST_SIZE))
    return idx[n:], idx[:n]


def positional_split(X, y, pos, seed):
    """Hold out a contiguous tail of each source sequence, so a test window's
    neighbours are never in training."""
    tr, te = [], []
    for src in np.unique(pos[:, 0]):
        m = np.flatnonzero(pos[:, 0] == src)
        order = m[np.argsort(pos[m, 1])]
        cut = int(round(len(order) * (1 - TEST_SIZE)))
        # drop a WINDOW-wide buffer so the two halves do not share a window
        tr.append(order[:cut - WINDOW])
        te.append(order[cut:])
    return np.concatenate(tr), np.concatenate(te)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--take", type=int, default=30000,
                    help="elements per sequence (memory bound)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="runs/audit/mavlink.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    raw = Path(a.raw)

    seqs = {}
    for f in sorted(raw.glob("*.npy")):
        seqs[f.stem.replace("_sequences", "")] = np.load(f)
    print("\nfiles")
    for k, v in seqs.items():
        print(f"  {k:<42} {len(v):>7,} message ids, {len(np.unique(v)):>3} distinct")

    old = {k[len("hitl_100000_"):]: v for k, v in seqs.items()
           if k.startswith("hitl_100000_")}
    if not all(n in old for p in PAIRS for n in p):
        raise SystemExit("the old paired captures were not found under --raw")

    results = []

    # ---- test 1: the split policy, on the task the paper reports ----------
    print("\ntest 1  binary normal-vs-attack, as the released pipeline builds it")
    X, y, pos = [], [], []
    for si, (name, seq) in enumerate(sorted(old.items())):
        w = windows(seq, a.take)
        X.append(w)
        y.append(np.full(len(w), 0 if name.startswith("normal") else 1))
        pos.append(np.stack([np.full(len(w), si), np.arange(len(w))], axis=1))
    X, y, pos = np.vstack(X), np.concatenate(y), np.vstack(pos)
    print(f"        {len(X):,} windows of {WINDOW}, overlap {WINDOW-1}/{WINDOW}")

    for seed in range(a.seeds):
        tr, te = random_split(X, y, seed)
        r = fit_predict(X[tr], y[tr], X[te], y[te], seed)
        tr2, te2 = positional_split(X, y, pos, seed)
        p = fit_predict(X[tr2], y[tr2], X[te2], y[te2], seed)
        results.append({"test": "binary", "seed": seed,
                        "random_window_split": r, "positional_split": p})
        log.info("  seed %d  random %.4f   positional %.4f   gap %+.4f",
                 seed, r, p, r - p)

    # ---- test 2: can two captures of the SAME class be told apart? --------
    print("\ntest 2  same-class capture discrimination "
          "(chance = 0.500; high means capture identity is readable)")
    for name_a, name_b in PAIRS:
        wa, wb = windows(old[name_a], a.take), windows(old[name_b], a.take)
        Xp = np.vstack([wa, wb])
        yp = np.concatenate([np.zeros(len(wa), int), np.ones(len(wb), int)])
        pp = np.vstack([
            np.stack([np.zeros(len(wa), int), np.arange(len(wa))], axis=1),
            np.stack([np.ones(len(wb), int), np.arange(len(wb))], axis=1)])
        rs, ps = [], []
        for seed in range(a.seeds):
            tr, te = random_split(Xp, yp, seed)
            rs.append(fit_predict(Xp[tr], yp[tr], Xp[te], yp[te], seed))
            tr2, te2 = positional_split(Xp, yp, pp, seed)
            ps.append(fit_predict(Xp[tr2], yp[tr2], Xp[te2], yp[te2], seed))
        results.append({"test": "same_class", "pair": f"{name_a} vs {name_b}",
                        "random_window_split": float(np.mean(rs)),
                        "positional_split": float(np.mean(ps))})
        print(f"  {name_a:<12} vs {name_b:<12} "
              f"random {np.mean(rs):.4f}   positional {np.mean(ps):.4f}")

    print("\n" + "=" * 72)
    b = [r for r in results if r["test"] == "binary"]
    print(f"binary task     random-window split  {np.mean([r['random_window_split'] for r in b]):.4f}")
    print(f"                positional split     {np.mean([r['positional_split'] for r in b]):.4f}")
    sc = [r for r in results if r["test"] == "same_class"]
    print(f"same-class      random-window split  "
          f"{np.mean([r['random_window_split'] for r in sc]):.4f}   (chance 0.500)")
    print(f"                positional split     "
          f"{np.mean([r['positional_split'] for r in sc]):.4f}")
    print("=" * 72)
    print("Two captures of the same class separating as well as normal from\n"
          "attack means the measured quantity is the capture, not the attack.")

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"window": WINDOW, "take": a.take,
                             "results": results}, indent=2) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
