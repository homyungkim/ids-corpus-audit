"""Artefacts promised in Sections 7.2 and 7.3 of the revised manuscript.

  release/data/hassler_sessions.csv
      one row per inferred recording session of the Hassler corpus: session id,
      class, instrumentation configuration, first and last source line
      (1-based lines of Dataset_T-ITS.csv), record count, start and end time.
  release/splits_corpora/hassler/loco_sessions.json
      leave-one-capture-out folds with recording sessions as captures.
  release/splits_corpora/hai/loco_merged.json
      leave-one-capture-out folds for HAI 21.03 with train1 and test3, which
      share 43,201 identical rows, merged into one capture.

The existing loco.UNDEFINED.json for the Hassler corpus is kept: at the level
of class blocks the policy remains undefined, and the file says so.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from rev_common import hassler_sessions, load  # noqa: E402
from rev_fusion import blocks, num, sessions  # noqa: E402

CONFIG = {0: "A", 2: "A", 4: "A", 6: "B", 8: "C"}


def session_table(path: str) -> list[dict]:
    starts = [i for i, r in enumerate(csv.reader(open(path, newline="")))
              if r and r[0] in ("timestamp_c", "timestamp_p", "mid")]
    rows, sid = [], 0
    for bi, x in enumerate(blocks(path)):
        if x["kind"] != "timestamp_c":
            continue
        lab = x["body"][0][x["head"].index("class")].strip()
        t = np.array([num(r[0]) for r in x["body"]])
        s = sessions(t)
        for k in np.unique(s):
            idx = np.where(s == k)[0]
            t0, t1 = float(t[idx[0]]), float(t[idx[-1]])
            if t0 > 1e9:
                fmt = lambda v: dt.datetime.fromtimestamp(v, dt.UTC).isoformat()
            else:
                fmt = lambda v: f"time of day {int(v // 3600):02d}:{int(v % 3600 // 60):02d}:{v % 60:06.3f}"
            rows.append({"session": f"s{sid:03d}", "class": lab, "configuration": CONFIG[bi],
                         "block": bi, "first_line": starts[bi] + 2 + int(idx[0]),
                         "last_line": starts[bi] + 2 + int(idx[-1]), "records": int(len(idx)),
                         "start": fmt(t0), "end": fmt(t1), "seconds": round(t1 - t0, 3)})
            sid += 1
    return rows


def loco(capture, y, corpus, basis):
    classes = np.unique(y)
    folds = []
    for c in np.unique(capture):
        te = capture == c
        usable = len(np.unique(y[~te])) == len(classes)
        folds.append({"held_out_capture": str(c), "usable": bool(usable),
                      "n_test": int(te.sum()),
                      "test_classes": sorted(set(map(str, y[te])))})
    return {"policy": "loco", "corpus": corpus, "basis": basis, "folds": folds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hassler", required=True)
    ap.add_argument("--hai", required=True)
    a = ap.parse_args()
    tab = session_table(a.hassler)
    out = ROOT / "release/data/hassler_sessions.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(tab[0]))
        w.writeheader()
        w.writerows(tab)
    print("wrote", out, len(tab), "sessions")

    r = load("hassler", a.hassler)
    cap = hassler_sessions(r, 60.0)
    spec = loco(cap, r.y, "hassler",
                "captures are recording sessions inferred from gaps of more "
                "than 60 s in the cyber timestamps (revision/rev_common.py); "
                "every class spans at least 2 sessions")
    p = ROOT / "release/splits_corpora/hassler/loco_sessions.json"
    p.write_text(json.dumps(spec, indent=1))
    print("wrote", p, sum(f["usable"] for f in spec["folds"]), "usable folds")

    h = load("hai", a.hai)
    cap = np.where(np.isin(h.capture, ["train1", "test3"]), "train1+test3",
                   h.capture).astype(object)
    spec = loco(cap, h.y, "hai",
                "train1 and test3 share 43,201 identical rows (13 July 2020, "
                "00:00-12:00) and are merged into one capture")
    p = ROOT / "release/splits_corpora/hai/loco_merged.json"
    p.write_text(json.dumps(spec, indent=1))
    print("wrote", p, sum(f["usable"] for f in spec["folds"]), "usable folds")


if __name__ == "__main__":
    main()
