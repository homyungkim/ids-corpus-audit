"""Hassler corpus without its two counter features (revision, Reviewer 3 (2)).

Permutation importance names frame.number (the packet's index in its capture
file) and wlan.seq (the 802.11 sequence number) as the features that identify
a recording session most strongly. Both are counters that grow within a
recording; neither describes the traffic. This script removes them and repeats
the split-cost measurement (all five models, all policies) and the fingerprint
index, so that the paper can say how much of the session effect the counters
carry.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rev_common import hassler_sessions, load  # noqa: E402
from rev_fingerprint import feature_names, identify, within_class  # noqa: E402
from rev_split_cost import run  # noqa: E402

COUNTERS = ("frame.number", "wlan.seq")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hassler", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = load("hassler", a.hassler)
    names = feature_names("hassler", a.hassler, r.X.shape[1])
    keep = [i for i, n in enumerate(names) if n not in COUNTERS]
    dropped = [n for n in names if n in COUNTERS]
    r2 = copy.copy(r)
    r2.X = r.X[:, keep]
    cap = hassler_sessions(r, 60.0)
    outp = Path(a.out)
    out = json.loads(outp.read_text()) if outp.exists() else {}
    out["dropped"] = dropped
    out["features_left"] = [names[i] for i in keep]
    fp = [identify(r2.X, cap, "rf", s) for s in (0, 1, 2)]
    out["fingerprint"] = {k: float(np.mean([f[k] for f in fp]))
                          for k in ("iota_acc", "iota_bal", "cohen", "ami")}
    out["within_class"] = within_class(r2.X, r2.y, cap)["weighted_iota_bal"]
    outp.write_text(json.dumps(out, indent=1))
    for _ in run("hassler_nocounters", r2, cap, Path(a.splits), out,
                 ["lr", "rf", "xgb", "mlp", "cnn"], split_key="hassler"):
        outp.write_text(json.dumps(out, indent=1))
    outp.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
