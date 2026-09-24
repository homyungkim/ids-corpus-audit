#!/usr/bin/env python3
"""What does a result on Dataset_T-ITS.csv actually measure?

The released corpus is five concatenated exports carrying ten header rows and
three incompatible schemas. `pandas.read_csv(path)` takes line 0 as the single
header, so the column it names `class` is index 37 -- a position that holds the
label in only three of the ten blocks. The other 21,672 rows come back with an
empty label and are removed by the `dropna()` that conventionally follows.

What survives that read is 33,102 rows, 37 cyber columns, and three classes.
Two of the four attack families named in the releasing paper's abstract are
gone, and so is every physical feature, whatever the directory is called.

This script trains one model family on the same corpus under three readings so
the reported numbers can be attributed to a task rather than to a method:

  naive-3        what read_csv + dropna leaves: 3 classes, cyber only
  correct-5      segment-aware: 5 classes, cyber features on the common schema
  correct-5-cp   segment-aware: 5 classes, cyber + physical

Classifiers are the ones this literature uses -- random forest, linear SVM,
and a small MLP -- so the comparison is against the field's own practice and
not against a strawman.

Splits are grouped by contiguous block so that adjacent, near-duplicate packet
rows cannot land on both sides; a random row split on this corpus inflates
every number reported here by a wide margin, which is its own finding.

Usage:
    python scripts/audit_benchmark.py --csv /path/to/Dataset_T-ITS.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

log = logging.getLogger("audit")

HEADER_STARTS = ("timestamp_c", "timestamp_p", "mid")

# The five families, normalised. The file itself is inconsistent: the cyber
# block of segment 1 says "DoS attack" where its physical block says "DoS".
FAMILY = {"benign": "benign", "dos attack": "DoS", "dos": "DoS",
          "replay": "Replay", "evil_twin": "evil_twin", "fdi": "FDI"}


# ---------------------------------------------------------------- structure
def segments(path: Path) -> list[dict]:
    rows = list(csv.reader(path.open(newline="")))
    hdr = [i for i, r in enumerate(rows) if r and r[0] in HEADER_STARTS]
    out = []
    for k, s in enumerate(hdr):
        e = hdr[k + 1] if k + 1 < len(hdr) else len(rows)
        head = [c.strip() for c in rows[s]]
        ci = [j for j, c in enumerate(head) if c.lower() == "class"]
        body = rows[s + 1:e]
        lab = FAMILY[Counter(r[ci[0]] for r in body
                             if len(r) > ci[0]).most_common(1)[0][0].strip().lower()]
        out.append({"index": k, "kind": "cyber" if head[0] == "timestamp_c" else "phys",
                    "header": head, "class_idx": ci[0], "rows": body,
                    "n": len(body), "label": lab, "line": s})
    return out


def _numeric(v: str) -> float:
    v = v.strip()
    if not v:
        return np.nan
    try:
        return float(v)
    except ValueError:
        # categorical (MAC address, IP, protocol string): hash to a stable code
        return float(abs(hash(v)) % 100_000)


def _matrix(rows, cols_idx) -> np.ndarray:
    return np.asarray([[_numeric(r[j]) if j < len(r) else np.nan
                        for j in cols_idx] for r in rows], dtype=np.float64)


# ---------------------------------------------------------------- readings
def read_naive(path: Path) -> dict:
    """Reproduce `pd.read_csv(path)` followed by `dropna()`."""
    rows = list(csv.reader(path.open(newline="")))
    head = [c.strip() for c in rows[0]]
    ci = head.index("class")
    keep, lab = [], []
    for r in rows[1:]:
        v = r[ci].strip() if len(r) > ci else ""
        if v.lower() in FAMILY:                 # non-empty, and a real label
            keep.append(r)
            lab.append(FAMILY[v.lower()])
    cols = [j for j in range(len(head)) if j != ci]
    X = _matrix(keep, cols)
    grp = np.asarray([i // 2000 for i in range(len(keep))])   # contiguous blocks
    return {"name": "naive-3", "X": X, "y": np.asarray(lab), "group": grp,
            "features": [head[j] for j in cols], "n_rows": len(keep)}


def read_correct(path: Path, with_physical: bool) -> dict:
    """Segment-aware read on the intersection of the per-segment schemas."""
    segs = segments(path)
    cy = [s for s in segs if s["kind"] == "cyber"]
    ph = [s for s in segs if s["kind"] == "phys"]

    def common(blocks):
        sets = [set(c for c in b["header"]
                    if c.lower() not in ("class",) and c) for b in blocks]
        return sorted(set.intersection(*sets))

    ccols = common(cy)
    pcols = common(ph) if with_physical else []

    X_all, y_all, g_all = [], [], []
    for gi, (c, p) in enumerate(zip(cy, ph)):
        xc = _matrix(c["rows"], [c["header"].index(n) for n in ccols])
        if with_physical:
            xp_src = _matrix(p["rows"], [p["header"].index(n) for n in pcols])
            # The two views have different lengths and no sample-level
            # correspondence. Stretch the shorter to the longer, preserving
            # order. The stretch factor is reported: it is the share of the
            # physical view that is interpolated rather than measured.
            n = len(xc)
            idx = np.clip(np.round(np.linspace(0, len(xp_src) - 1, n)).astype(int),
                          0, len(xp_src) - 1)
            x = np.hstack([xc, xp_src[idx]])
        else:
            x = xc
        X_all.append(x)
        y_all.append(np.full(len(x), c["label"], dtype=object))
        g_all.append(np.full(len(x), gi * 100 + np.arange(len(x)) // 2000))

    return {"name": "correct-5-cp" if with_physical else "correct-5",
            "X": np.vstack(X_all), "y": np.concatenate(y_all),
            "group": np.concatenate(g_all),
            "features": ccols + pcols, "n_rows": sum(len(a) for a in X_all)}


# ---------------------------------------------------------------- evaluation
def evaluate(ds: dict, seed: int = 0, split: str = "grouped") -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, g = ds["X"], ds["y"], ds["group"]
    if split == "grouped":
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                        random_state=seed).split(X, y, g))
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(X))
        tr, te = train_test_split(idx, test_size=0.3, random_state=seed,
                                  stratify=y)
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                               random_state=seed),
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=400, n_jobs=-1)),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=120, random_state=seed)),
    }
    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])

    out = []
    for name, m in models.items():
        t0 = time.time()
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        out.append({"reading": ds["name"], "model": name, "split": split,
                    "accuracy": float(accuracy_score(y[te], p)),
                    "macro_f1": float(f1_score(y[te], p, average="macro")),
                    "per_class_f1": {str(c): float(v) for c, v in zip(
                        sorted(set(y[te])),
                        f1_score(y[te], p, average=None,
                                 labels=sorted(set(y[te]))))},
                    "n_train": int(len(tr)), "n_test": int(len(te)),
                    "n_classes": int(len(set(y))), "n_features": int(X.shape[1]),
                    "seconds": round(time.time() - t0, 1)})
        log.info("  %-14s %-12s %-7s acc %.4f  macroF1 %.4f",
                 ds["name"], name, split, out[-1]["accuracy"],
                 out[-1]["macro_f1"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/audit/benchmark.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    path = Path(a.csv)

    segs = segments(path)
    print("\nsegment structure")
    print(f"{'#':>2} {'kind':<6} {'label':<10} {'rows':>7} {'features':>9} {'header line':>12}")
    for s in segs:
        print(f"{s['index']:>2} {s['kind']:<6} {s['label']:<10} {s['n']:>7} "
              f"{len(s['header'])-1:>9} {s['line']:>12}")

    naive = read_naive(path)
    c5 = read_correct(path, with_physical=False)
    c5cp = read_correct(path, with_physical=True)

    # The naive read keeps 37 cyber columns because it only ever sees the three
    # blocks that share that schema. The corpus as a whole cannot supply 37
    # columns for five families -- segments 4 and 5 were captured with a
    # different instrument. To keep the task effect separable from the feature
    # effect, repeat the naive reading restricted to the same 14 columns the
    # five-family reading is limited to.
    keep = [i for i, f in enumerate(naive["features"]) if f in set(c5["features"])]
    naive14 = dict(naive, name="naive-3 (14f)", X=naive["X"][:, keep],
                   features=[naive["features"][i] for i in keep])

    readings = [naive, naive14, c5, c5cp]

    print("\nwhat each reading yields")
    print(f"{'reading':<14} {'rows':>8} {'features':>9} {'classes':>8}   families")
    for d in readings:
        fam = sorted(set(d["y"].tolist()))
        print(f"{d['name']:<14} {d['n_rows']:>8,} {d['X'].shape[1]:>9} "
              f"{len(fam):>8}   {', '.join(fam)}")

    rows = []
    print()
    for d in readings:
        for sp in ("grouped", "random"):
            rows += evaluate(d, a.seed, sp)

    print("\n" + "=" * 74)
    print(f"{'reading':<15}{'cls':>4}{'feat':>5}  {'model':<13}"
          f"{'grouped':>10}{'random':>10}{'leak':>9}")
    print("-" * 74)
    seen = set()
    for r in rows:
        k = (r["reading"], r["model"])
        if k in seen:
            continue
        seen.add(k)
        gv = next(x["accuracy"] for x in rows if (x["reading"], x["model"]) == k
                  and x["split"] == "grouped")
        rv = next(x["accuracy"] for x in rows if (x["reading"], x["model"]) == k
                  and x["split"] == "random")
        print(f"{r['reading']:<15}{r['n_classes']:>4}{r['n_features']:>5}  "
              f"{r['model']:<13}{gv:>10.4f}{rv:>10.4f}{rv-gv:>+9.4f}")
    print("=" * 74)

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "csv": str(path),
        "segments": [{k: v for k, v in s.items() if k != "rows" and k != "header"}
                     for s in segs],
        "results": rows}, indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
