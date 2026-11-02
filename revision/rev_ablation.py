"""Do adapter decisions move the results? (revision, Reviewer 1 comment 8)

Every adapter makes choices the corpus does not dictate: how many records to
keep, how long a window is, how far windows advance. Each choice is varied
here, one at a time, around the published setting, and the two quantities the
paper's argument rests on are re-measured with a random forest:

  iota_bal   capture-balanced fingerprint index (rev_fingerprint)
  capture    capture cost: random-row balanced accuracy minus pooled
             leave-one-capture-out balanced accuracy

The published setting is listed first for each corpus.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import (ADAPTERS, fit_predict, hassler_sessions,  # noqa: E402
                        hassler_timestamps, metrics)
from rev_fingerprint import identify  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("abl")

VARIANTS = {
    "hassler": [{"max_records": 40000}, {"max_records": 20000},
                {"max_records": 40000, "_gap": 30},
                {"max_records": 40000, "_gap": 120},
                {"max_records": 40000, "_gap": 300}],
    "mavlink": [{"window": 128}, {"window": 64}, {"window": 256},
                {"max_records": 20000}],
    "shadow": [{"window": 64, "stride": 32}, {"window": 32, "stride": 16},
               {"window": 128, "stride": 64}, {"max_records": 20000}],
    "uavcan": [{"max_records": 60000}, {"max_records": 30000},
               {"max_records": 120000}],
    "lumi": [{"max_records": 60000}, {"max_records": 30000},
             {"max_records": 120000}],
    "hai": [{"max_records": 60000}, {"max_records": 30000},
            {"max_records": 120000}],
}


def measure(r, cap):
    fp = identify(r.X, cap, "rf", 0)
    classes = np.unique(r.y)
    rr = []
    for s in (0, 1, 2):
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(r))
        te, tr = np.sort(perm[:int(0.3 * len(r))]), np.sort(perm[int(0.3 * len(r)):])
        rr.append(metrics(r.y[te], fit_predict("rf", r.X, r.y, tr, te, s),
                          classes)["bal"])
    pred = np.empty(len(r), dtype=object)
    done = np.zeros(len(r), bool)
    for i, c in enumerate(np.unique(cap)):
        te, tr = np.where(cap == c)[0], np.where(cap != c)[0]
        if len(np.unique(r.y[tr])) < len(classes):
            continue
        pred[te] = fit_predict("rf", r.X, r.y, tr, te, i)
        done[te] = True
    loco = metrics(r.y[done], pred[done], classes)["bal"] if done.any() else None
    per = {str(c): int(len(np.unique(cap[r.y == c]))) for c in classes}
    return {"iota_bal": fp["iota_bal"], "iota_acc": fp["iota_acc"],
            "random_bal": float(np.mean(rr)), "loco_bal": loco,
            "capture_cost": None if loco is None else float(np.mean(rr) - loco),
            "kappa": int(min(per.values())), "rows": int(len(r)),
            "captures": int(len(np.unique(cap)))}


def main():
    ap = argparse.ArgumentParser()
    for k in VARIANTS:
        ap.add_argument(f"--{k}")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    outp = Path(a.out)
    out = json.loads(outp.read_text()) if outp.exists() else {}
    for key, vs in VARIANTS.items():
        path = getattr(a, key)
        if not path:
            continue
        for v in vs:
            tag = f"{key}:{json.dumps(v, sort_keys=True)}"
            if tag in out:
                continue
            kw = {k: x for k, x in v.items() if not k.startswith("_")}
            r = ADAPTERS[key](path, **kw)
            if key == "hassler":
                r.views = {"hassler_timestamps":
                           hassler_timestamps(path, kw.get("max_records", 40000))}
                cap = hassler_sessions(r, v.get("_gap", 60.0))
            else:
                cap = r.capture
            out[tag] = measure(r, cap)
            log.info("%-45s %s", tag, {k: (round(x, 3) if isinstance(x, float)
                                            else x) for k, x in out[tag].items()})
            outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
