#!/usr/bin/env python3
"""Does the corpus support the cyber-physical fusion claim it was built for?

The releasing paper's contribution is that fusing cyber and physical UAV
features improves intrusion detection. Testing that on the released file runs
into two obstacles, and this script measures both.

Obstacle 1 -- coverage. The five physical blocks share exactly three feature
names (pitch, roll, yaw). Segments 1-3 carry a 16-feature Tello schema;
segment 4 carries a different 21-feature SDK schema; segment 5 carries 31
fields including estimator and control signals. A five-family fusion
experiment is therefore capped at three physical features, and segment 5
exposes `mpitch/mroll/myaw` alongside `pitch/roll/yaw`, so even those three are
not obviously commensurable across segments. The first three segments, in
contrast, share a full 37-feature cyber and 16-feature physical schema, so a
clean three-family fusion experiment is available -- and is the one the naive
read makes invisible by discarding the physical blocks.

Obstacle 2 -- correspondence. Within a segment the cyber and physical blocks
have different lengths and no sample-level pairing:

    benign  9,425 cyber vs 4,290 physical   ->  2.2x stretch
    DoS    11,671 cyber vs   973 physical   -> 12.0x stretch
    Replay 12,006 cyber vs   973 physical   -> 12.3x stretch

Pairing them requires a rule the experimenter invents. For DoS and Replay the
rule has to synthesise roughly 92% of the physical samples. Any gain measured
after such a pairing may come from the pairing rather than from the data, so
the gain is reported against two nulls. Every arm applies the same monotone
stretch, so all three share the duplication structure and the train/test
disjointness of the physical block; only what the stretch is applied to
changes.

Arms:
    cyber        37 cyber features, no physical   (what the naive read leaves)
    cyber+phys   37 cyber + 16 physical, order-paired
    null: order  the class's own physical rows, permuted before the stretch.
                 Destroys cyber-physical ordering, keeps everything else.
                 A gain that survives this was never sample-level correspondence.
    null: class  each class given a DIFFERENT class's physical block. A gain
                 that survives this is block identity -- which capture session
                 a row came from -- and not an attack-specific signature.

Usage:
    python scripts/audit_fusion.py --csv /path/to/Dataset_T-ITS.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_benchmark import _matrix, segments          # noqa: E402

log = logging.getLogger("fusion")


def build(path: Path, mode: str, seed: int = 0) -> dict:
    """mode: 'cyber' | 'paired' | 'null'."""
    segs = segments(path)
    cy = [s for s in segs if s["kind"] == "cyber"][:3]     # benign, DoS, Replay
    ph = [s for s in segs if s["kind"] == "phys"][:3]

    ccols = sorted(set.intersection(*[
        set(c for c in b["header"] if c and c.lower() != "class") for b in cy]))
    pcols = sorted(set.intersection(*[
        set(c for c in b["header"] if c and c.lower() != "class") for b in ph]))
    # timestamps are bookkeeping, not signal, and leak block identity
    ccols = [c for c in ccols if c != "timestamp_c"]
    pcols = [c for c in pcols if c != "timestamp_p"]

    rng = np.random.default_rng(seed)
    # for null_class, rotate the physical blocks among the classes
    rot = list(range(len(ph)))
    if mode == "null_class":
        rot = rot[1:] + rot[:1]
    ph = [ph[i] for i in rot]

    X, y, g, stretch = [], [], [], []
    for gi, (c, p) in enumerate(zip(cy, ph)):
        xc = _matrix(c["rows"], [c["header"].index(n) for n in ccols])
        n = len(xc)
        if mode == "cyber":
            x = xc
        else:
            xp = _matrix(p["rows"], [p["header"].index(n_) for n_ in pcols])
            # Every arm uses the SAME monotone stretch, so every arm has the
            # same duplication structure and the same disjointness between the
            # train and test halves of the physical block. Only what the
            # stretch is applied to changes. An earlier version drew the null
            # by sampling physical rows uniformly, which let a physical row
            # land on both sides of the split and handed the null an advantage
            # that had nothing to do with the question.
            base = np.clip(np.round(np.linspace(0, len(xp) - 1, n)).astype(int),
                           0, len(xp) - 1)
            if mode == "paired":
                src = xp
            elif mode == "null_order":
                # same rows, same multiplicities, cyber-physical ordering
                # destroyed: isolates sample-level correspondence
                src = xp[rng.permutation(len(xp))]
            else:                                 # null_class: block rotated above
                src = xp
            x = np.hstack([xc, src[base]])
            stretch.append({"class": c["label"], "cyber": n, "physical": len(xp),
                            "stretch": round(n / len(xp), 1),
                            "synthesised_share": round(1 - len(xp) / n, 3)})
        X.append(x)
        y.append(np.full(n, c["label"], dtype=object))
        g.append(gi * 1000 + np.arange(n) // 2000)

    return {"name": {"cyber": "cyber", "paired": "cyber+phys",
                     "null_order": "null: order", "null_class": "null: class"}[mode],
            "X": np.vstack(X), "y": np.concatenate(y), "group": np.concatenate(g),
            "n_features": len(ccols) + (0 if mode == "cyber" else len(pcols)),
            "stretch": stretch}


def score(ds: dict, seed: int = 0) -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, g = ds["X"], ds["y"], ds["group"]
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=seed).split(X, y, g))
    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
    out = []
    for name, m in {
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                               random_state=seed),
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=400)),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=150, random_state=seed)),
    }.items():
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        out.append({"arm": ds["name"], "model": name,
                    "accuracy": float(accuracy_score(y[te], p)),
                    "macro_f1": float(f1_score(y[te], p, average="macro")),
                    "n_features": ds["n_features"]})
        log.info("  %-16s %-13s acc %.4f  macroF1 %.4f",
                 ds["name"], name, out[-1]["accuracy"], out[-1]["macro_f1"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="runs/audit/fusion.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    path = Path(a.csv)

    d0 = build(path, "paired", 0)
    print("\npairing required within each segment")
    print(f"{'class':<9}{'cyber rows':>12}{'physical rows':>15}{'stretch':>10}{'synthesised':>13}")
    for s in d0["stretch"]:
        print(f"{s['class']:<9}{s['cyber']:>12,}{s['physical']:>15,}"
              f"{s['stretch']:>9.1f}x{100*s['synthesised_share']:>12.1f}%")

    rows = []
    for seed in range(a.seeds):
        for mode in ("cyber", "paired", "null_order", "null_class"):
            rows += [dict(r, seed=seed) for r in score(build(path, mode, seed), seed)]

    print("\n" + "=" * 78)
    print(f"{'model':<14}{'cyber':>10}{'cyber+phys':>13}{'gain':>9}"
          f"{'null:order':>13}{'null:class':>13}")
    print("-" * 78)
    summary = []
    for model in ("RandomForest", "LogReg", "MLP"):
        def mu(arm):
            v = [r["accuracy"] for r in rows
                 if r["model"] == model and r["arm"] == arm]
            return float(np.mean(v)) if v else float("nan")
        c, p = mu("cyber"), mu("cyber+phys")
        no, nc = mu("null: order"), mu("null: class")
        summary.append({"model": model, "cyber": c, "paired": p,
                        "null_order": no, "null_class": nc,
                        "gain_paired": p - c, "gain_null_order": no - c,
                        "gain_null_class": nc - c})
        print(f"{model:<14}{c:>10.4f}{p:>13.4f}{p-c:>+9.4f}"
              f"{no:>13.4f}{nc:>13.4f}")
    print("=" * 78)
    print("null:order  physical rows permuted, then the same stretch  -> tests"
          " sample-level correspondence")
    print("null:class  each class given another class's physical block -> tests"
          " block-identity leakage")

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"stretch": d0["stretch"], "runs": rows,
                               "summary": summary}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
