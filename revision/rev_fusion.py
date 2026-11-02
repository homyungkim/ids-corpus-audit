"""Label-independent controls for the cross-layer fusion result (revision).

Responds to Reviewer 2 (7). In the published experiment (scripts/audit_fusion.py)
each cyber row is paired with physical rows drawn from the physical block of
its own class, for test rows as well as training rows. The pairing therefore
uses the label, and the gain it produces cannot be attributed to capture
identity rather than to label information.

This script keeps the published arms and adds two pairings that never consult
a label:

  time      each cyber row is paired with the physical row nearest to it in
            time, searched over the pooled physical rows of all three blocks
            (both views carry a seconds-of-day clock in segments 0-2)
  random    each cyber row is paired with a physical row drawn uniformly from
            the pooled physical rows

Evaluation holds out whole recording sessions (inferred from timestamp gaps,
rev_common.hassler_sessions logic applied to the full cyber block), five
seeds, and reports accuracy and balanced accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("fusion")


def blocks(path):
    rows = list(csv.reader(Path(path).open(newline="")))
    starts = [i for i, r in enumerate(rows)
              if r and r[0] in ("timestamp_c", "timestamp_p", "mid")]
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(rows)
        h = [c.strip() for c in rows[s]]
        while h and h[-1] == "":
            h.pop()
        body = [r[:len(h)] for r in rows[s + 1:e]]
        out.append({"head": h, "body": body, "kind": h[0]})
    return out


def num(v):
    try:
        return float(v)
    except ValueError:
        return np.nan


def sessions(t, gap=60.0, min_rows=50):
    brk = np.concatenate([[True], (np.diff(t) > gap) | (np.diff(t) < 0)])
    seg = np.cumsum(brk) - 1
    sizes = np.bincount(seg)
    last, remap = None, {}
    for s in range(len(sizes)):
        remap[s] = last if (sizes[s] < min_rows and last is not None) else s
        if remap[s] == s:
            last = s
    return np.array([remap[s] for s in seg])


def build(path):
    b = blocks(path)
    cy = [x for x in b if x["kind"] == "timestamp_c"][:3]
    ph = [x for x in b if x["kind"] == "timestamp_p"][:3]
    ccols = sorted(set.intersection(*[set(c for c in x["head"]
                   if c and c != "class" and not c.startswith("timestamp"))
                   for x in cy]))
    pcols = sorted(set.intersection(*[set(c for c in x["head"]
                   if c and c.lower() != "class"
                   and not c.startswith("timestamp")) for x in ph]))
    C, Ct, Cy, Cs, P, Pt, Pb = [], [], [], [], [], [], []
    sid_off = 0
    for i, (c, p) in enumerate(zip(cy, ph)):
        ci = [c["head"].index(n) for n in ccols]
        lab = c["body"][0][c["head"].index("class")].strip().lower()
        t = np.array([num(r[0]) for r in c["body"]])
        C.append(np.array([[num(r[j]) for j in ci] for r in c["body"]]))
        Ct.append(t)
        Cy.append(np.full(len(t), lab.replace(" attack", ""), dtype=object))
        s = sessions(t)
        Cs.append(s + sid_off)
        sid_off += s.max() + 1
        pi = [p["head"].index(n) for n in pcols]
        P.append(np.array([[num(r[j]) for j in pi] for r in p["body"]]))
        Pt.append(np.array([num(r[0]) for r in p["body"]]))
        Pb.append(np.full(len(p["body"]), i))
    return {"C": C, "Ct": Ct, "Cy": Cy, "Cs": Cs, "P": P, "Pt": Pt, "Pb": Pb,
            "ccols": ccols, "pcols": pcols}


def arm(d, mode, rng):
    """Return the feature matrix for one arm; labels are never consulted
    except by the published class-block arms, which are kept for reference."""
    Xc = np.vstack(d["C"])
    if mode == "cyber":
        return Xc
    Pall, Ptall = np.vstack(d["P"]), np.concatenate(d["Pt"])
    if mode == "time":
        o = np.argsort(Ptall)
        ts = Ptall[o]
        q = np.concatenate(d["Ct"])
        j = np.clip(np.searchsorted(ts, q), 1, len(ts) - 1)
        j = np.where(np.abs(ts[j - 1] - q) <= np.abs(ts[j] - q), j - 1, j)
        return np.hstack([Xc, Pall[o][j]])
    if mode == "random":
        return np.hstack([Xc, Pall[rng.integers(0, len(Pall), len(Xc))]])
    # published arms: class-block monotone stretch
    parts = []
    nb = len(d["C"])
    rot = list(range(nb))
    if mode == "null_class":
        rot = rot[1:] + rot[:1]
    for i in range(nb):
        xc, xp = d["C"][i], d["P"][rot[i]]
        base = np.clip(np.round(np.linspace(0, len(xp) - 1, len(xc)))
                       .astype(int), 0, len(xp) - 1)
        src = xp[rng.permutation(len(xp))] if mode == "null_order" else xp
        parts.append(np.hstack([xc, src[base]]))
    return np.vstack(parts)


def evaluate(X, y, g, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=seed).split(X, y, g))
    # every class must be present on both sides
    m = RandomForestClassifier(n_estimators=150, n_jobs=2, random_state=seed)
    m.fit(np.nan_to_num(X[tr]), y[tr])
    p = m.predict(np.nan_to_num(X[te]))
    return float(accuracy_score(y[te], p)), \
        float(balanced_accuracy_score(y[te], p)), \
        sorted(set(y[te].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hassler", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = build(a.hassler)
    y = np.concatenate(d["Cy"])
    g = np.concatenate(d["Cs"])
    log.info("cyber rows %d, sessions %d, cyber cols %d, physical cols %d",
             len(y), len(np.unique(g)), len(d["ccols"]), len(d["pcols"]))
    res = {"sessions_per_class": {str(c): int(len(np.unique(g[y == c])))
                                  for c in np.unique(y)}, "arms": {}}
    for mode in ("cyber", "class_block", "null_order", "null_class", "time",
                 "random"):
        vals = []
        seed = 0
        while len(vals) < 5 and seed < 50:
            rng = np.random.default_rng(seed)
            X = arm(d, "paired" if mode == "class_block" else mode, rng)
            acc, bal, present = evaluate(X, y, g, seed)
            seed += 1
            if len(present) < len(np.unique(y)):
                continue           # a split that drops a class is not usable
            vals.append((acc, bal))
        res["arms"][mode] = {"acc": float(np.mean([v[0] for v in vals])),
                             "acc_sd": float(np.std([v[0] for v in vals])),
                             "bal": float(np.mean([v[1] for v in vals])),
                             "bal_sd": float(np.std([v[1] for v in vals])),
                             "n": len(vals), "seeds_tried": seed}
        log.info("%-12s acc %.4f  bal %.4f (n=%d)", mode,
                 res["arms"][mode]["acc"], res["arms"][mode]["bal"], len(vals))
    # how far apart in time are time-paired rows?
    Ptall = np.concatenate(d["Pt"])
    q = np.concatenate(d["Ct"])
    o = np.sort(Ptall)
    j = np.clip(np.searchsorted(o, q), 1, len(o) - 1)
    dist = np.minimum(np.abs(o[j - 1] - q), np.abs(o[j] - q))
    pb = np.concatenate(d["Pb"])[np.argsort(Ptall)]
    jj = np.where(np.abs(o[j - 1] - q) <= np.abs(o[j] - q), j - 1, j)
    cls_idx = {c: i for i, c in enumerate(["benign", "dos", "replay"])}
    same = np.mean(pb[jj] == np.array([cls_idx.get(v, -1) for v in y]))
    res["time_pairing"] = {"median_abs_dt_s": float(np.median(dist)),
                           "p90_abs_dt_s": float(np.percentile(dist, 90)),
                           "share_partner_from_same_class_block": float(same)}
    log.info("time pairing: median |dt| %.2fs, same-block partner %.3f",
             np.median(dist), same)
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
