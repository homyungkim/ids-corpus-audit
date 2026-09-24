#!/usr/bin/env python3
"""Structural audit of the HCRL UAVCAN (DroneCAN) attack dataset.

    ocslab.hksecurity.net/Datasets/uavcan-attack-dataset
    IEEE DataPort doi:10.21227/fcyc-bb14
    Technical report: arXiv:2212.09268

The frame counts in this corpus are exactly as documented -- all ten files
match the technical report's table to the frame. What does not match is the
format. Both the report and the DataPort record describe a CSV with six
columns (Label, Timestamp, Interface, CAN ID, DLC, Data); the files are named
`.bin`, carry no header, use no commas, and are SocketCAN candump text with a
label word prepended:

    Normal (000.000000)  can0  10015501   [8]  00 00 00 00 08 00 00 C0

The field count varies row to row, because the data field holds DLC bytes.
`pandas.read_csv` on these files does not fail -- it returns a single column
of strings, or, with a whitespace separator, a ragged frame whose columns mean
different things on different rows.

This script parses them correctly and then measures four things that bear on
what a result from this corpus means:

  1. format conformance: can every line be parsed, and against what grammar
  2. time: is the timestamp monotone within a capture, and what does the
     per-capture reset imply for concatenation
  3. attack labelling: which scenarios mix attack types, and how much of the
     attack mass carries no type label
  4. difficulty: how far a trivial rule gets, under a split that does not let
     a frame's neighbours answer for it

Usage:
    python scripts/audit_uavcan.py --dir /path/to/extracted
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import numpy as np

log = logging.getLogger("uavcan")

# Normal (000.000000)  can0  10015501   [8]  00 00 00 00 08 00 00 C0
LINE = re.compile(
    r"^(?P<label>\w+)\s+\((?P<ts>[\d.]+)\)\s+(?P<iface>\S+)\s+"
    r"(?P<canid>[0-9A-Fa-f]+)\s+\[(?P<dlc>\d+)\]\s*(?P<data>[0-9A-Fa-f ]*)$")

# Scenario table from the technical report (arXiv:2212.09268)
SCENARIO = {
    1: ("Flooding", 0.0015, 180), 2: ("Flooding", 0.005, 180),
    3: ("Fuzzy", 0.0015, 180), 4: ("Fuzzy", 0.005, 180),
    5: ("Fuzzy", 0.005, 210), 6: ("Replay", 0.005, 280),
    7: ("Flooding & Fuzzy", 0.005, 240), 8: ("Fuzzy & Replay", 0.005, 240),
    9: ("Fuzzy & Replay", 0.005, 270), 10: ("Flooding, Fuzzy & Replay", 0.005, 220),
}


def parse(path: Path) -> dict:
    lab, ts, canid, dlc, nbytes = [], [], [], [], []
    bad, widths = [], Counter()
    with path.open("rb") as fh:
        for ln, raw in enumerate(fh, 1):
            s = raw.decode("ascii", "replace").rstrip("\r\n")
            if not s:
                continue
            widths[len(s.split())] += 1
            m = LINE.match(s)
            if not m:
                if len(bad) < 5:
                    bad.append((ln, s[:90]))
                continue
            lab.append(m["label"])
            ts.append(float(m["ts"]))
            canid.append(int(m["canid"], 16))
            dlc.append(int(m["dlc"]))
            nbytes.append(len(m["data"].split()))
    return {"label": np.array(lab), "ts": np.array(ts),
            "canid": np.array(canid, dtype=np.int64),
            "dlc": np.array(dlc), "nbytes": np.array(nbytes),
            "bad": bad, "field_widths": dict(widths)}


def trivial_rule_accuracy(d: dict, seed: int = 0) -> dict:
    """How far does a rule on one field get, split so neighbours cannot help?

    The split holds out a contiguous tail of the capture: a held-out frame's
    immediate predecessors are all in the training half, which is the mildest
    honest policy available on a single ordered capture.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    y = (d["label"] == "Attack").astype(int)
    n = len(y)
    cut = int(n * 0.7)
    tr, te = np.arange(cut), np.arange(cut, n)

    # inter-arrival time and the raw fields; no windowing, one frame at a time
    dt = np.diff(d["ts"], prepend=d["ts"][0])
    X = np.stack([d["canid"], d["dlc"], dt], axis=1)

    out = {}
    # (a) a CAN ID never seen among training-half normal frames
    seen = set(d["canid"][tr][y[tr] == 0].tolist())
    pred = np.array([c not in seen for c in d["canid"][te]], dtype=int)
    out["unseen_can_id"] = float(accuracy_score(y[te], pred))

    # (b) one threshold on inter-arrival time
    best = 0.0
    for q in np.quantile(dt[tr], np.linspace(0.01, 0.99, 60)):
        for flip in (0, 1):
            p = (dt[te] < q).astype(int) if flip == 0 else (dt[te] >= q).astype(int)
            best = max(best, accuracy_score(y[te], p))
    out["inter_arrival_threshold"] = float(best)

    # (c) a small forest on the three fields
    m = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed,
                               max_depth=12)
    m.fit(X[tr], y[tr])
    out["forest_3_fields"] = float(accuracy_score(y[te], m.predict(X[te])))

    out["attack_share_in_test"] = float(y[te].mean())
    out["majority_baseline"] = float(max(y[te].mean(), 1 - y[te].mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="runs/audit/uavcan.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    root = Path(a.dir)

    rows = []
    print("\nformat conformance and timing")
    print(f"{'file':<18}{'frames':>10}{'unparsed':>9}{'field widths':>14}"
          f"{'ts span (s)':>13}{'monotone':>10}{'dlc=bytes':>11}")
    print("-" * 85)
    parsed = {}
    for i in range(1, 11):
        p = root / f"type{i}_label.bin"
        if not p.exists():
            log.warning("%s missing", p.name)
            continue
        d = parse(p)
        parsed[i] = d
        mono = bool(np.all(np.diff(d["ts"]) >= 0))
        agree = bool(np.all(d["dlc"] == d["nbytes"]))
        span = d["ts"][-1] - d["ts"][0]
        widths = ",".join(str(k) for k in sorted(d["field_widths"]))
        print(f"{p.name:<18}{len(d['label']):>10,}{len(d['bad']):>9}"
              f"{widths:>14}{span:>13.3f}{str(mono):>10}{str(agree):>11}")
        rows.append({"scenario": i, "attack": SCENARIO[i][0],
                     "frames": int(len(d["label"])),
                     "normal": int((d["label"] == "Normal").sum()),
                     "attack_frames": int((d["label"] == "Attack").sum()),
                     "unparsed": len(d["bad"]),
                     "field_widths": d["field_widths"],
                     "ts_span_s": float(span),
                     "ts_monotone": mono, "dlc_matches_bytes": agree,
                     "documented_seconds": SCENARIO[i][2]})

    print("\nlabelling: which scenarios carry more than one attack type")
    multi = [r for r in rows if "&" in r["attack"]]
    tot_a = sum(r["attack_frames"] for r in rows)
    mul_a = sum(r["attack_frames"] for r in multi)
    for r in rows:
        mark = "  <- two or more types, no per-frame type label" if "&" in r["attack"] else ""
        print(f"  scenario {r['scenario']:>2}  {r['attack']:<26}"
              f"{r['attack_frames']:>8,} attack frames{mark}")
    print(f"\n  attack frames in multi-type captures: {mul_a:,} of {tot_a:,} "
          f"({100*mul_a/tot_a:.1f}%)")
    print("  The label column is binary. Attack type lives only in the file "
          "name and the report's table,\n  so for these captures no frame can "
          "be assigned to a type.")

    print("\nbenign-only material")
    print(f"  captures with zero attack frames: "
          f"{sum(1 for r in rows if r['attack_frames'] == 0)} of {len(rows)}")
    print("  Every normal frame in this corpus was recorded during a capture "
          "in which an attack ran.")

    print("\ndifficulty, held-out tail of each capture (no windowing)")
    print(f"{'scenario':<10}{'majority':>10}{'unseen CAN id':>15}"
          f"{'inter-arrival':>15}{'forest(3 fields)':>18}")
    print("-" * 70)
    diff = []
    for i, d in parsed.items():
        r = trivial_rule_accuracy(d)
        diff.append(dict(r, scenario=i))
        print(f"{i:<10}{r['majority_baseline']:>10.4f}{r['unseen_can_id']:>15.4f}"
              f"{r['inter_arrival_threshold']:>15.4f}{r['forest_3_fields']:>18.4f}")
    print("-" * 70)
    print(f"{'mean':<10}{np.mean([r['majority_baseline'] for r in diff]):>10.4f}"
          f"{np.mean([r['unseen_can_id'] for r in diff]):>15.4f}"
          f"{np.mean([r['inter_arrival_threshold'] for r in diff]):>15.4f}"
          f"{np.mean([r['forest_3_fields'] for r in diff]):>18.4f}")

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"files": rows, "difficulty": diff}, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
