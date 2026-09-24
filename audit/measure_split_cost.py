#!/usr/bin/env python3
"""Measure what each reference split policy costs, per corpus.

Reads the splits written by make_reference_splits.py and reports accuracy
under each of them for two classifiers. The quantity of interest is not the
accuracy but the distance between policies: a corpus where the random-row
number and the grouped number agree has little neighbour leakage, and one
where they diverge has a great deal. Running both a tree ensemble and a
network guards against the gap being a property of one model family.

    python audit/measure_split_cost.py \
        --hassler_release release/data/v1.0.0 \
        --splits release/splits_corpora \
        --out runs/audit/split_cost.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import ADAPTERS                              # noqa: E402
from checklist import Records                              # noqa: E402
from make_reference_splits import blocks_of, load_digest   # noqa: E402

MODELS = ("rf", "mlp")


def fit(X, y, tr, te, kind: str, seed: int) -> tuple[float, float]:
    """Return (accuracy, balanced accuracy) for one fit.

    Accuracy is what published work on these corpora reports, so it is the
    figure that makes the comparison against that work meaningful. It is also
    useless on a corpus whose majority class covers 99 % of the records, and
    one of the six is in that position, so the balanced figure is carried
    alongside it rather than instead of it.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                   random_state=seed)
    else:
        # early stopping rather than a fixed iteration budget: at 300 fixed
        # iterations the optimiser had not converged on any policy, and a
        # number read off an unconverged fit says as much about the budget
        # as about the split
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(64, 32),
                                        max_iter=1000, early_stopping=True,
                                        n_iter_no_change=15,
                                        random_state=seed))
    # integer labels: the network's internal validation split rejects the
    # object-dtype label arrays the adapters produce
    classes, enc = np.unique(y, return_inverse=True)
    m.fit(X[tr], enc[tr])
    pred = m.predict(X[te])
    return (float(accuracy_score(enc[te], pred)),
            float(balanced_accuracy_score(enc[te], pred)))


def random_row_idx(n: int, test_size: float, seed: int):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(n * test_size))
    te = np.sort(perm[:k])
    return np.setdiff1d(np.arange(n), te), te


def run_corpus(key: str, r: Records, sdir: Path) -> dict:
    man = json.loads((sdir / key / "manifest.json").read_text())
    live = load_digest(r)
    if live != man["load_digest"]:
        raise SystemExit(
            f"{key}: this load does not match the one the splits were built "
            f"from ({live} vs {man['load_digest']}). Regenerate the splits, "
            f"or load the corpus the way the manifest describes.")

    out = {"corpus": key, "name": r.name, "rows": int(len(r)),
           "load_digest": live, "policies": {}}
    ts = man["test_size"]

    def record(policy, scores):
        out["policies"][policy] = {
            m: {"mean": float(np.mean([a for a, _ in v])),
                "sd": float(np.std([a for a, _ in v])),
                "bal_mean": float(np.mean([b for _, b in v])),
                "bal_sd": float(np.std([b for _, b in v])),
                "n": len(v),
                "runs": [round(a, 4) for a, _ in v],
                "bal_runs": [round(b, 4) for _, b in v]}
            for m, v in scores.items()}
        line = "  ".join(
            f"{m} {np.mean([a for a,_ in v]):.4f} (bal {np.mean([b for _,b in v]):.4f})"
            for m, v in scores.items())
        logging.info("  %-14s %s", policy, line)

    # random row
    sc = {m: [] for m in MODELS}
    for s in man["seeds"]:
        tr, te = random_row_idx(len(r), ts, s)
        for m in MODELS:
            sc[m].append(fit(r.X, r.y, tr, te, m, s))
    record("random-row", sc)

    # grouped
    for policy, meta in man["policies"].items():
        if not policy.startswith("grouped-"):
            continue
        b = int(policy.split("-")[1])
        g = blocks_of(r, b)
        sc = {m: [] for m in MODELS}
        for s in man["seeds"]:
            spec = json.loads((sdir / key / f"{policy}__seed{s}.json").read_text())
            held = set(spec["held_out_groups"])
            mask = np.array([str(x) in held for x in g])
            tr, te = np.where(~mask)[0], np.where(mask)[0]
            if len(np.unique(r.y[tr])) < len(np.unique(r.y)):
                continue
            for m in MODELS:
                sc[m].append(fit(r.X, r.y, tr, te, m, s))
        if sc[MODELS[0]]:
            record(policy, sc)

    # leave one capture out
    f = sdir / key / "loco.json"
    if f.exists():
        folds = [x for x in json.loads(f.read_text())["folds"] if x["usable"]]
        sc = {m: [] for m in MODELS}
        for i, fold in enumerate(folds):
            te = np.where(r.capture == fold["held_out_capture"])[0]
            tr = np.where(r.capture != fold["held_out_capture"])[0]
            for m in MODELS:
                sc[m].append(fit(r.X, r.y, tr, te, m, i))
        record("loco", sc)
    else:
        out["policies"]["loco"] = {"defined": False}
        logging.info("  %-14s undefined on this corpus", "loco")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ADAPTERS:
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--splits", default="release/splits_corpora")
    ap.add_argument("--out", default="runs/audit/split_cost.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    sdir = Path(a.splits)

    res = []
    for key, fn in ADAPTERS.items():
        path = getattr(a, key)
        if not path:
            continue
        if not (sdir / key / "manifest.json").exists():
            logging.warning("%s: no splits under %s; run "
                            "make_reference_splits.py first", key, sdir)
            continue
        logging.info("%s", key)
        res.append(run_corpus(key, fn(path), sdir))

    if not res:
        print("nothing measured")
        return 1

    print("\n" + "=" * 86)
    print(f"{'corpus':<16}{'policy':<14}"
          + "".join(f"{m.upper()+' acc/bal':>14}" for m in MODELS)
          + f"{'gap acc | bal':>20}")
    print("-" * 86)
    for c in res:
        base = c["policies"].get("random-row")
        for pol, v in c["policies"].items():
            if not isinstance(v, dict) or "defined" in v:
                print(f"{c['corpus']:<16}{pol:<14}"
                      + " " * (14 * len(MODELS))
                      + f"{'undefined':>18}")
                continue
            cells = "".join(f"{v[m]['mean']:>7.4f}/{v[m]['bal_mean']:.3f}"
                            for m in MODELS)
            if pol == "random-row":
                gap = ""
            else:
                gap = " ".join(f"{v[m]['mean']-base[m]['mean']:+.3f}/"
                               f"{v[m]['bal_mean']-base[m]['bal_mean']:+.3f}"
                               for m in MODELS)
            print(f"{c['corpus']:<16}{pol:<14}{cells}{gap:>18}")
        print("-" * 86)
    print("=" * 86)

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
