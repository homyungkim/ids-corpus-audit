"""Timing evidence for two claims the reviewers questioned (revision).

Reviewer 2 (6): in the UAVCAN releases, does every normal frame occur while an
attack is running, and how do class proportions vary between captures?
Reviewer 3 (4): can a held-out capture overlap in time with training captures?

For every capture this reports its time span and, for the two UAVCAN releases,
the proportion of attack frames, the time of the first and last attack frame,
how many normal frames fall before the first or after the last attack frame,
and how many normal frames fall in one-second windows that contain no attack
frame at all. It then lists every pair of captures whose time spans overlap.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

CANDUMP = re.compile(r"^(?P<label>\w+)\s+\((?P<ts>[\d.]+)\)")


def uavcan_2022(d):
    out = {}
    for i in range(1, 11):
        p = Path(d) / f"type{i}_label.bin"
        lab, ts = [], []
        for raw in p.open("rb"):
            m = CANDUMP.match(raw.decode("ascii", "replace"))
            if m:
                lab.append(m["label"].lower())
                ts.append(float(m["ts"]))
        out[f"scenario-{i:02d}"] = (np.array(lab), np.array(ts))
    return out


def lumi(d):
    out = {}
    for p in sorted(Path(d).glob("*.csv")):
        lab, ts = [], []
        with p.open(newline="") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                if len(row) >= 5:
                    lab.append(row[0].lower())
                    ts.append(float(row[1]))
        out[p.stem] = (np.array(lab), np.array(ts))
    return out


def class_timing(caps):
    rep = {}
    for c, (lab, ts) in caps.items():
        a = lab != "normal"
        ta = ts[a]
        first, last = ta.min(), ta.max()
        nrm = ~a
        before = int((nrm & (ts < first)).sum())
        after = int((nrm & (ts > last)).sum())
        sec = np.floor(ts - ts.min()).astype(int)
        attack_secs = set(np.unique(sec[a]).tolist())
        quiet = np.array([s not in attack_secs for s in sec])
        rep[c] = {"frames": int(len(lab)), "attack_share": float(a.mean()),
                  "span_s": float(ts.max() - ts.min()),
                  "first_attack_s": float(first - ts.min()),
                  "last_attack_s": float(last - ts.min()),
                  "normal_before_first_attack": before,
                  "normal_after_last_attack": after,
                  "normal_in_attack_free_seconds": int((nrm & quiet).sum()),
                  "normal_total": int(nrm.sum()),
                  "t0": float(ts.min()), "t1": float(ts.max())}
    return rep


def spans_hai(d):
    out = {}
    for p in sorted(Path(d).glob("*.csv")):
        with p.open(newline="") as f:
            r = csv.reader(f)
            head = next(r)
            first = next(r)[0]
            last = first
            for row in r:
                last = row[0]
        out[p.stem] = (first.strip(), last.strip())
    return out


def spans_shadow(d):
    from scapy.all import PcapNgReader
    out = {}
    for p in sorted(Path(d).glob("*.pcapng")):
        t0 = t1 = None
        with PcapNgReader(str(p)) as r:
            for x in r:
                t = float(x.time)
                t0 = t if t0 is None else t0
                t1 = t
        out[p.stem] = (t0, t1)
    return out


def overlaps(spans):
    keys = sorted(spans)
    pairs = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            (a0, a1), (b0, b1) = spans[a], spans[b]
            ov = min(a1, b1) - max(a0, b0)
            if ov > 0:
                pairs.append({"a": a, "b": b, "overlap": ov})
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uavcan")
    ap.add_argument("--lumi")
    ap.add_argument("--hai")
    ap.add_argument("--shadow")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    if a.uavcan:
        t = class_timing(uavcan_2022(a.uavcan))
        res["uavcan"] = {"captures": t, "overlaps": overlaps(
            {k: (v["t0"], v["t1"]) for k, v in t.items()})}
    if a.lumi:
        t = class_timing(lumi(a.lumi))
        res["lumi"] = {"captures": t, "overlaps": overlaps(
            {k: (v["t0"], v["t1"]) for k, v in t.items()})}
    if a.hai:
        import datetime as dt
        sp = spans_hai(a.hai)
        num = {k: (dt.datetime.fromisoformat(v[0]).timestamp(),
                   dt.datetime.fromisoformat(v[1]).timestamp())
               for k, v in sp.items()}
        res["hai"] = {"spans": sp, "overlaps": overlaps(num)}
    if a.shadow:
        sp = spans_shadow(a.shadow)
        res["shadow"] = {"spans": sp, "overlaps": overlaps(sp)}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, "overlapping capture pairs:", len(v["overlaps"]))


if __name__ == "__main__":
    main()
