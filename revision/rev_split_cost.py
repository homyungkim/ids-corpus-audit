"""Split-policy cost, measured the same way on every corpus (revision).

Responds to Reviewer 2 (comments 1-3), Reviewer 1 (2, 3), Reviewer 3 (8) and
Reviewer 8 (2). Differences from audit/measure_split_cost.py:

* one protocol per policy, used on every corpus:
    random-row     30 % of records, five seeds
    grouped-B      held-out contiguous runs of B records (published splits)
    loco           every capture held out once (leave-one-capture-out)
  The Hassler corpus has no session field; its captures are recording
  sessions inferred from timestamp gaps (rev_common.hassler_sessions).
* balanced accuracy for loco is computed once, from the pooled out-of-fold
  predictions, never averaged over folds; per-class recall and per-capture
  accuracy are stored beside it;
* five model families with hyperparameters fixed in advance and identical
  across policies: logistic regression, random forest, XGBoost, MLP, 1-D CNN;
* uncertainty: seed spread for random/grouped, a capture-level (cluster)
  bootstrap for the pooled loco figures.

Results are written after every (corpus, policy, model) so a long run can be
resumed.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import fit_predict, hassler_sessions, load, metrics  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "audit"))
from make_reference_splits import blocks_of  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rev")

MODELS = ("lr", "rf", "xgb", "mlp", "cnn")
SEEDS = (0, 1, 2, 3, 4)
TEST = 0.3


def random_row(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    k = int(round(n * TEST))
    te = np.sort(perm[:k])
    return np.setdiff1d(np.arange(n), te), te


def cluster_boot(y, p, cap, B=1000, seed=0):
    """95 % interval for pooled balanced accuracy, resampling captures."""
    rng = np.random.default_rng(seed)
    caps = np.unique(cap)
    idx_of = {c: np.where(cap == c)[0] for c in caps}
    vals = []
    for _ in range(B):
        pick = rng.choice(caps, len(caps), replace=True)
        ii = np.concatenate([idx_of[c] for c in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(metrics(y[ii], p[ii])["bal"])
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run(key, r, capture, splits_dir, out, models, loco_only=False, split_key=None):
    res = out.setdefault(key, {"rows": int(len(r)),
                               "captures": int(len(np.unique(capture))),
                               "classes": {str(c): int((r.y == c).sum())
                                           for c in np.unique(r.y)},
                               "policies": {}})
    pol = res["policies"]
    classes = np.unique(r.y)
    per = {str(c): int(len(np.unique(capture[r.y == c]))) for c in classes}
    res["captures_per_class"] = per
    res["kappa"] = int(min(per.values()))

    split_key = split_key or key
    for m in models:
        # random row
        k = f"random-row/{m}"
        if k not in pol and not loco_only:
            t0 = time.time()
            runs = []
            for s in SEEDS:
                tr, te = random_row(len(r), s)
                p = fit_predict(m, r.X, r.y, tr, te, s)
                runs.append(metrics(r.y[te], p, classes))
            pol[k] = _summ(runs, time.time() - t0)
            log.info("%-8s %-22s acc %.4f bal %.4f", key, k,
                     pol[k]["acc_mean"], pol[k]["bal_mean"])
            yield
        # grouped, from the published split files (same held-out groups)
        for B in (500, 3000):
            k = f"grouped-{B}/{m}"
            if k in pol or loco_only:
                continue
            t0 = time.time()
            g = blocks_of(r, B)
            runs = []
            for s in SEEDS:
                spec = json.loads((splits_dir / split_key /
                                   f"grouped-{B}__seed{s}.json").read_text())
                held = set(spec["held_out_groups"])
                mask = np.array([str(x) in held for x in g])
                tr, te = np.where(~mask)[0], np.where(mask)[0]
                if len(np.unique(r.y[tr])) < len(classes):
                    continue
                p = fit_predict(m, r.X, r.y, tr, te, s)
                runs.append(metrics(r.y[te], p, classes))
            pol[k] = _summ(runs, time.time() - t0)
            log.info("%-8s %-22s acc %.4f bal %.4f", key, k,
                     pol[k]["acc_mean"], pol[k]["bal_mean"])
            yield
        # leave one capture out, pooled
        k = f"loco/{m}"
        if k in pol:
            continue
        t0 = time.time()
        pred = np.empty(len(r), dtype=object)
        done = np.zeros(len(r), bool)
        fold_acc, skipped = {}, []
        for i, c in enumerate(np.unique(capture)):
            te = np.where(capture == c)[0]
            tr = np.where(capture != c)[0]
            if len(np.unique(r.y[tr])) < len(classes):
                skipped.append(str(c))
                continue
            pred[te] = fit_predict(m, r.X, r.y, tr, te, i)
            done[te] = True
            fold_acc[str(c)] = float(np.mean(pred[te] == r.y[te]))
        if not done.any():
            pol[k] = {"defined": False, "skipped": skipped}
        else:
            yy, pp, cc = r.y[done], pred[done], capture[done]
            mm = metrics(yy, pp, classes)
            pol[k] = {"defined": True, "acc_mean": mm["acc"],
                      "bal_mean": mm["bal"], "recall": mm["recall"],
                      "bal_ci95": cluster_boot(yy, pp, cc),
                      "fold_acc": fold_acc,
                      "fold_acc_sd": float(np.std(list(fold_acc.values()))),
                      "n_folds": len(fold_acc), "skipped": skipped,
                      "coverage": float(done.mean()),
                      "seconds": round(time.time() - t0, 1)}
        log.info("%-8s %-22s %s", key, k,
                 f"acc {pol[k]['acc_mean']:.4f} bal {pol[k]['bal_mean']:.4f} "
                 f"ci {pol[k]['bal_ci95']}" if pol[k].get("defined")
                 else "undefined")
        yield


def _summ(runs, sec):
    a = [x["acc"] for x in runs]
    b = [x["bal"] for x in runs]
    rec = {}
    for x in runs:
        for c, v in x["recall"].items():
            rec.setdefault(c, []).append(v)
    return {"acc_mean": float(np.mean(a)), "acc_sd": float(np.std(a)),
            "bal_mean": float(np.mean(b)), "bal_sd": float(np.std(b)),
            "recall": {c: float(np.mean(v)) for c, v in rec.items()},
            "n": len(runs), "seconds": round(sec, 1)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("hassler", "mavlink", "uavcan", "lumi", "shadow", "hai"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--gap", type=float, default=60.0)
    a = ap.parse_args()
    outp = Path(a.out)
    out = json.loads(outp.read_text()) if outp.exists() else {}
    models = a.models.split(",")
    for key in ("hassler", "mavlink", "hai", "uavcan", "lumi", "shadow"):
        path = getattr(a, key)
        if not path:
            continue
        r = load(key, path)
        cap = hassler_sessions(r, a.gap) if key == "hassler" else r.capture
        log.info("%s: %d records, %d captures", key, len(r),
                 len(np.unique(cap)))
        for _ in run(key, r, cap, Path(a.splits), out, models):
            outp.write_text(json.dumps(out, indent=1))
        if key == "hai":
            # train1 and test3 share 43,201 identical rows (12 h); treat the
            # pair as one capture so no held-out row has a copy in training
            cap2 = np.where(np.isin(cap, ["train1", "test3"]),
                            "train1+test3", cap).astype(object)
            for _ in run("hai_merged", r, cap2, Path(a.splits), out, models,
                         loco_only=True, split_key="hai"):
                outp.write_text(json.dumps(out, indent=1))
    outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
