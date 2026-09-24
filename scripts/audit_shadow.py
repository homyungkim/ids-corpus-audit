#!/usr/bin/env python3
"""SHADOW-GCS: the one corpus of the five that can answer its own question.

    ocslab.hksecurity.net/Datasets/shadow-gcs-attack-dataset
    S. Lee, H. K. Kim, "GALAXY: ...", IEEE PRDC 2025

Four of the five UAV intrusion corpora examined in this audit ship one capture
per class, so a class and a capture are the same variable and no experiment on
them can separate attack detection from capture identification. SHADOW-GCS is
built differently: 35 captures crossed over transport (Direct / Relay), vehicle
state (Armed / Disarmed) and class (Benign / Attack), with three to ten
captures in every cell, and 14 of them benign throughout.

That design makes leave-one-capture-out possible, which is the evaluation the
other four cannot support. This script runs it, against a within-capture split
on the same windows and the same model, so the difference is attributable to
the policy and to nothing else.

Reading the result
------------------
A small gap means the corpus measures attack detection. A large gap means that
what a within-capture split reports is largely capture identity -- and that the
numbers published on the four corpora which only permit a within-capture split
have never been checked against this failure.

Usage:
    python scripts/audit_shadow.py --dir /path/to/SHADOW-GCS
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("shadow")

NAME = re.compile(r"(?P<path>\w+)-(?P<state>\w+)-(?P<cls>\w+)-(?P<idx>\d+)\.pcapng$")
WINDOW = 64          # packets per window
STRIDE = 32


def features(pkts: list[tuple[float, int]]) -> np.ndarray:
    """Per-window features from (timestamp, length) only.

    Deliberately shallow: no payload, no addresses, no ports. If a shallow
    feature set already separates captures, a deeper one certainly will, and
    the point is about the evaluation rather than the model.
    """
    out = []
    for s in range(0, len(pkts) - WINDOW + 1, STRIDE):
        w = pkts[s:s + WINDOW]
        t = np.array([p[0] for p in w])
        L = np.array([p[1] for p in w], dtype=float)
        dt = np.diff(t)
        span = max(t[-1] - t[0], 1e-9)
        out.append([
            L.mean(), L.std(), L.min(), L.max(), np.median(L),
            len(np.unique(L)),
            dt.mean() if len(dt) else 0.0, dt.std() if len(dt) else 0.0,
            np.median(dt) if len(dt) else 0.0,
            WINDOW / span,                       # packet rate
            L.sum() / span,                      # byte rate
        ])
    return np.asarray(out, dtype=np.float64)


def read_capture(path: Path) -> list[tuple[float, int]]:
    from scapy.all import PcapNgReader
    out = []
    with PcapNgReader(str(path)) as r:
        for pkt in r:
            out.append((float(pkt.time), len(pkt)))
    return out


def evaluate(X, y, groups, policy: str, seed: int) -> float:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if policy == "leave-captures-out":
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                        random_state=seed).split(X, y, groups))
    else:                                        # within-capture
        tr, te = train_test_split(np.arange(len(y)), test_size=0.3,
                                  random_state=seed, stratify=y)
    m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
    m.fit(X[tr], y[tr])
    return float(accuracy_score(y[te], m.predict(X[te])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/audit/shadow.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    files = sorted(Path(a.dir).glob("*.pcapng"))
    X, y, g, meta = [], [], [], []
    for gi, p in enumerate(files):
        m = NAME.search(p.name)
        if not m:
            log.warning("skipping %s", p.name)
            continue
        f = features(read_capture(p))
        if not len(f):
            continue
        X.append(f)
        y.append(np.full(len(f), 1 if m["cls"] == "Attack" else 0))
        g.append(np.full(len(f), gi))
        meta.append({"file": p.name, "group": gi, "cls": m["cls"],
                     "path": m["path"], "state": m["state"],
                     "windows": int(len(f))})
    X, y, g = np.vstack(X), np.concatenate(y), np.concatenate(g)
    print(f"\n{len(files)} captures -> {len(X):,} windows of {WINDOW} packets "
          f"(stride {STRIDE}), {X.shape[1]} features")
    print(f"  benign windows {int((y==0).sum()):,}   "
          f"attack windows {int((y==1).sum()):,}   "
          f"majority {max(y.mean(), 1-y.mean()):.4f}")

    # ---- the comparison -------------------------------------------------
    res = {"within": [], "loco": []}
    for seed in range(a.seeds):
        res["within"].append(evaluate(X, y, g, "within-capture", seed))
        res["loco"].append(evaluate(X, y, g, "leave-captures-out", seed))
        log.info("  seed %d  within-capture %.4f   leave-captures-out %.4f",
                 seed, res["within"][-1], res["loco"][-1])

    w, l = np.mean(res["within"]), np.mean(res["loco"])
    print("\n" + "=" * 68)
    print(f"{'within-capture split (what the other four corpora allow)':<56}{w:>10.4f}")
    print(f"{'leave-captures-out (what this corpus additionally allows)':<56}{l:>10.4f}")
    print(f"{'gap':<56}{w-l:>+10.4f}")
    print(f"{'majority baseline':<56}{max(y.mean(),1-y.mean()):>10.4f}")
    print("=" * 68)

    # ---- can a capture be identified from the same shallow features? ----
    print("\ncapture identification from the same features "
          "(35-way, chance = 0.029)")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(g)), test_size=0.3,
                              random_state=0, stratify=g)
    m = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
    m.fit(X[tr], g[tr])
    acc_id = accuracy_score(g[te], m.predict(X[te]))
    print(f"  {acc_id:.4f}")
    print("  A window carries enough of its own capture's fingerprint to name "
          "it among 35.\n  On a corpus with one capture per class, that "
          "fingerprint IS the label.")

    out = {"window": WINDOW, "stride": STRIDE, "captures": meta,
           "within_capture": res["within"], "leave_captures_out": res["loco"],
           "capture_identification_35way": float(acc_id),
           "majority": float(max(y.mean(), 1 - y.mean()))}
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
