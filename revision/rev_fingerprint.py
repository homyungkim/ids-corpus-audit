"""Validation of the capture-fingerprint index (revision).

Responds to Reviewer 1 (1, 6), Reviewer 2 (5), Reviewer 3 (2, 6) and
Reviewer 8 (2, 3). For each corpus the capture-identification task of check C4
is repeated under variations of every design choice the reviewers named:

  normalisation  iota_acc  (published): accuracy rescaled by 1/|C|
                 iota_bal  capture-balanced accuracy rescaled by 1/|C|,
                           so a large capture cannot carry the score
                 cohen     chance-corrected agreement with chance from the
                           empirical capture marginals
                 ami       adjusted mutual information between predicted and
                           true capture (information-theoretic, chance = 0)
  control        within-class index: captures are identified only among
                 captures of the same class, so class information cannot
                 contribute (defined only where a class spans >= 2 captures)
  classifier     logistic regression, k-NN, random forest, XGBoost
  representation adapter features; the same features with the three most
                 important ones removed; for MAVLink, a message-id histogram
                 per window instead of the raw window
  sampling       minimum capture size 0 / 10 / 100 / 500; every capture
                 downsampled to the size of the smallest
  explanation    permutation importance of each feature for capture
                 identification (random forest)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import hassler_sessions, load, make_model  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fp")

SHADOW_NAMES = ["len_mean", "len_std", "len_min", "len_max", "len_median",
                "len_distinct", "iat_mean", "iat_std", "iat_median",
                "pkt_rate", "byte_rate"]
CAN_NAMES = ["can_id", "dlc", "inter_arrival"]


def feature_names(key, path, d):
    if key in ("uavcan", "lumi"):
        return CAN_NAMES
    if key == "shadow":
        return SHADOW_NAMES
    if key == "mavlink":
        return [f"pos{i}" for i in range(d)]
    if key == "hai":
        f = sorted(Path(path).glob("*.csv"))[0]
        head = next(csv.reader(open(f, newline="")))
        return [c.strip() for c in head
                if c.strip() != "time" and not c.strip().startswith("attack")]
    if key == "hassler":
        rows = list(csv.reader(Path(path).open(newline="")))
        starts = [i for i, r in enumerate(rows)
                  if r and r[0] in ("timestamp_c", "timestamp_p", "mid")]
        cy = []
        for i in starts:
            h = [c.strip() for c in rows[i]]
            while h and h[-1] == "":
                h.pop()
            if h[0] == "timestamp_c":
                cy.append(set(c for c in h if c and c != "class"
                              and not c.startswith("timestamp")))
        return sorted(set.intersection(*cy))
    return [f"f{i}" for i in range(d)]


def mavlink_hist(X):
    ids = np.unique(X.astype(int))
    return np.stack([(X == v).sum(1) for v in ids], 1).astype(float), \
        [f"msgid_{v}" for v in ids]


def identify(X, cap, kind, seed=0, min_size=10, equalize=False):
    """Capture identification on a stratified 70/30 split. Returns scores."""
    from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score,
                                 cohen_kappa_score, recall_score)
    from sklearn.model_selection import train_test_split
    rng = np.random.default_rng(seed)
    caps, cnt = np.unique(cap, return_counts=True)
    keepc = caps[cnt >= min_size]
    idx = np.where(np.isin(cap, keepc))[0]
    if equalize:
        m = min(cnt[cnt >= min_size])
        idx = np.concatenate([rng.choice(np.where(cap == c)[0], m,
                                         replace=False) for c in keepc])
    if len(keepc) < 2:
        return None
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed,
                              stratify=cap[idx])
    model = make_model(kind, seed) if kind != "knn" else _knn()
    model.fit(np.nan_to_num(X[tr]), cap[tr])
    p = model.predict(np.nan_to_num(X[te]))
    k = len(keepc)
    acc = accuracy_score(cap[te], p)
    bal = float(np.mean(recall_score(cap[te], p, labels=keepc, average=None,
                                     zero_division=0)))
    return {"k": int(k), "n": int(len(idx)),
            "iota_acc": (acc - 1 / k) / (1 - 1 / k),
            "iota_bal": (bal - 1 / k) / (1 - 1 / k),
            "cohen": float(cohen_kappa_score(cap[te], p)),
            "ami": float(adjusted_mutual_info_score(cap[te], p)),
            "acc": float(acc), "bal": bal,
            "majority": float(cnt[np.isin(caps, keepc)].max()
                              / cnt[np.isin(caps, keepc)].sum())}


def _knn():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), KNeighborsClassifier(5, n_jobs=2))


def within_class(X, y, cap, kind="rf", seeds=(0, 1, 2)):
    """Capture identification restricted to the captures of one class."""
    out = {}
    for c in np.unique(y):
        m = y == c
        if len(np.unique(cap[m])) < 2:
            out[str(c)] = None
            continue
        vals = [identify(X[m], cap[m], kind, s) for s in seeds]
        vals = [v for v in vals if v]
        out[str(c)] = {"k": vals[0]["k"], "n": int(m.sum()),
                       "iota_bal": float(np.mean([v["iota_bal"] for v in vals])),
                       "iota_acc": float(np.mean([v["iota_acc"] for v in vals]))}
    defined = {c: v for c, v in out.items() if v}
    if defined:
        w = np.array([v["n"] for v in defined.values()], float)
        agg = float(np.average([v["iota_bal"] for v in defined.values()],
                               weights=w))
    else:
        agg = None
    return {"per_class": out, "weighted_iota_bal": agg}


def importance(X, cap, names, seed=0, top=8):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(np.arange(len(cap)), test_size=0.3,
                              random_state=seed, stratify=cap)
    m = RandomForestClassifier(n_estimators=150, n_jobs=2, random_state=seed)
    m.fit(np.nan_to_num(X[tr]), cap[tr])
    sub = te[:min(len(te), 6000)]
    pi = permutation_importance(m, np.nan_to_num(X[sub]), cap[sub],
                                n_repeats=5, random_state=seed, n_jobs=1)
    order = np.argsort(-pi.importances_mean)
    return [{"feature": names[i], "drop_in_acc": float(pi.importances_mean[i]),
             "sd": float(pi.importances_std[i])} for i in order[:top]], order


def analyse(key, path, r, cap, label):
    log.info("== %s (%s): %d records, %d captures", key, label, len(r),
             len(np.unique(cap)))
    X, y = r.X, r.y
    names = feature_names(key, path, X.shape[1])
    res = {"captures": int(len(np.unique(cap)))}
    # baseline, published settings, three seeds
    base = [identify(X, cap, "rf", s) for s in (0, 1, 2)]
    res["base_rf"] = {k: float(np.mean([b[k] for b in base]))
                      for k in ("iota_acc", "iota_bal", "cohen", "ami", "acc",
                                "bal", "majority")}
    log.info("  base rf  iota_acc %.3f iota_bal %.3f cohen %.3f ami %.3f",
             *[res["base_rf"][k] for k in ("iota_acc", "iota_bal", "cohen",
                                           "ami")])
    # classifiers
    res["by_model"] = {}
    for kind in ("lr", "knn", "xgb"):
        v = identify(X, cap, kind, 0)
        res["by_model"][kind] = v
        log.info("  %-4s     iota_acc %.3f iota_bal %.3f", kind,
                 v["iota_acc"], v["iota_bal"])
    # sampling
    res["min_size"] = {}
    for ms in (0, 10, 100, 500):
        v = identify(X, cap, "rf", 0, min_size=ms)
        res["min_size"][str(ms)] = v and {k: v[k] for k in
                                          ("k", "n", "iota_acc", "iota_bal")}
    v = identify(X, cap, "rf", 0, equalize=True)
    res["equalized"] = v and {k: v[k] for k in ("k", "n", "iota_acc",
                                                "iota_bal")}
    log.info("  equalized iota_acc %.3f iota_bal %.3f",
             res["equalized"]["iota_acc"], res["equalized"]["iota_bal"])
    # within class
    res["within_class"] = within_class(X, y, cap)
    log.info("  within-class weighted iota_bal %s",
             res["within_class"]["weighted_iota_bal"])
    # explanation and reduced representation
    rep, order = importance(X, cap, names)
    res["importance"] = rep
    log.info("  top features %s", [(t["feature"], round(t["drop_in_acc"], 3))
                                   for t in rep[:5]])
    if X.shape[1] > 3:
        keep = np.setdiff1d(np.arange(X.shape[1]), order[:3])
        v = identify(X[:, keep], cap, "rf", 0)
        res["reduced_minus_top3"] = {k: v[k] for k in ("iota_acc", "iota_bal")}
    if key == "mavlink":
        H, hn = mavlink_hist(X)
        v = identify(H, cap, "rf", 0)
        res["histogram_repr"] = {k: v[k] for k in ("iota_acc", "iota_bal")}
        rep2, _ = importance(H, cap, hn)
        res["importance_histogram"] = rep2
        res["within_class_histogram"] = within_class(H, y, cap)
        log.info("  histogram repr iota_bal %.3f; within-class %s",
                 v["iota_bal"],
                 res["within_class_histogram"]["weighted_iota_bal"])
    return res


def main():
    ap = argparse.ArgumentParser()
    for k in ("hassler", "mavlink", "uavcan", "lumi", "shadow", "hai"):
        ap.add_argument(f"--{k}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outp = Path(a.out)
    out = json.loads(outp.read_text()) if outp.exists() else {}
    for key in ("hassler", "mavlink", "hai", "uavcan", "lumi", "shadow"):
        path = getattr(a, key)
        if not path:
            continue
        r = load(key, path)
        if key == "hassler":
            for label, cap in (("block", r.capture),
                               ("session", hassler_sessions(r, 60.0))):
                k2 = f"hassler_{label}"
                if k2 not in out:
                    out[k2] = analyse(key, path, r, cap, label)
                    outp.write_text(json.dumps(out, indent=1))
        elif key not in out:
            out[key] = analyse(key, path, r, r.capture, "file")
            outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
