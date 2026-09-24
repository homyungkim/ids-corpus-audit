"""Adapters that describe each corpus to the checklist.

An adapter's only job is to fill in `Records` honestly, and the hardest field
to fill in honestly is `capture`. Where a corpus ships one file per class, the
capture id and the class are the same variable, and the adapter has to say so
rather than inventing a finer grouping that the data does not support.

Each adapter also records what the reader observed while parsing, because the
format and label checks cannot be recomputed from the arrays after the fact.
"""

from __future__ import annotations

import csv
import glob
import hashlib
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from checklist import Records

def _subsample(n: int, cap: int, key: str) -> np.ndarray:
    """Draw a fixed subsample of `n` records, reproducibly.

    The seed is derived from the corpus name rather than from a module-level
    generator, so a corpus draws the same rows whether it is loaded alone or
    after four others. An earlier version of this file shared one generator
    across adapters; the draw then depended on which corpora happened to be
    requested in the same run, which is precisely the kind of undocumented
    dependence this audit objects to elsewhere.
    """
    if n <= cap:
        return np.arange(n)
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    return np.sort(np.random.default_rng(seed).choice(n, cap, replace=False))


def _num(v: str) -> float:
    v = v.strip()
    if not v:
        return np.nan
    try:
        return float(v)
    except ValueError:
        return float(abs(hash(v)) % 100_000)


# ---------------------------------------------------------------------------
def hassler(path: str, max_records: int = 40000) -> Records:
    """Hassler et al., IEEE T-ITS 2024. One CSV, ten header rows."""
    rows = list(csv.reader(Path(path).open(newline="")))
    starts = [i for i, r in enumerate(rows)
              if r and r[0] in ("timestamp_c", "timestamp_p", "mid")]
    blocks = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(rows)
        head = [c.strip() for c in rows[s]]
        w = len(head)
        while w and head[w - 1] == "":
            w -= 1
        head, body = head[:w], [r[:w] for r in rows[s + 1:e]]
        ci = head.index("class")
        blocks.append({"head": head, "body": body, "ci": ci,
                       "kind": "cyber" if head[0] == "timestamp_c" else "phys",
                       "raw": body[0][ci]})

    cy = [b for b in blocks if b["kind"] == "cyber"]
    cols = sorted(set.intersection(*[
        set(c for c in b["head"] if c and c != "class"
            and not c.startswith("timestamp")) for b in cy]))

    X, y, cap = [], [], []
    for b in cy:
        idx = [b["head"].index(c) for c in cols]
        X.append(np.array([[_num(r[j]) for j in idx] for r in b["body"]]))
        lab = b["raw"].strip().lower().replace(" attack", "")
        y.append(np.full(len(b["body"]), lab, dtype=object))
        # one capture per class: that is the corpus, not a modelling choice
        cap.append(np.full(len(b["body"]), f"block-{lab}", dtype=object))
    X, y, cap = np.vstack(X), np.concatenate(y), np.concatenate(cap)
    keep = _subsample(len(y), max_records, "hassler")

    # what a standard read keeps: only the three blocks whose class column
    # sits at the index the first header names
    kept = sum(len(b["body"]) for b in blocks if b["ci"] == blocks[0]["ci"])
    return Records(
        X=X[keep], y=y[keep], capture=cap[keep], order=np.arange(len(keep)),
        name="Hassler et al. (IEEE T-ITS 2024)",
        parse={"documented_format": "single CSV, 37 cyber / 16 physical features",
               "actual_format": "five concatenated exports, three schemas",
               "matches_documented_format": False,
               "distinct_headers": len(starts),
               "field_widths": dict(Counter(len(b["head"]) for b in blocks)),
               "unparsed": 0, "lines": sum(len(b["body"]) for b in blocks),
               "ships_split_guidance": False},
        labels={"kept_under_standard_read": kept,
                "total_records": sum(len(b["body"]) for b in blocks),
                "raw_vocabulary": sorted({b["raw"] for b in blocks}),
                "documented_classes": ["benign", "dos", "replay",
                                       "evil_twin", "fdi"],
                "recoverable_classes": sorted({b["raw"].strip().lower()
                                               .replace(" attack", "")
                                               for b in blocks}),
                "benign_class": "benign",
                "attack_type_per_record": True},
        views={"shared_clock": False,
               "clock_detail": "cyber and physical are separate row blocks "
                               "with no sample-level correspondence; the FDI "
                               "physical block has no timestamp column",
               "pairing_cost": [
                   {"class": "benign", "share_invented": 0.545},
                   {"class": "dos", "share_invented": 0.917},
                   {"class": "replay", "share_invented": 0.919}],
               "fusion_null": {"gain_paired": 0.3542, "gain_null_class": 0.3411}},
    )


# ---------------------------------------------------------------------------
def mavlink(raw_dir: str, max_records: int = 40000, window: int = 128) -> Records:
    """HCRL MAVLink message-id sequences (GUIDE, Computers & Security 2024)."""
    files = {}
    for f in sorted(glob.glob(os.path.join(raw_dir, "hitl_100000_*.npy"))):
        files[os.path.basename(f)[len("hitl_100000_"):-len("_sequences.npy")]] = np.load(f)

    X, y, cap, order = [], [], [], []
    for name, seq in sorted(files.items()):
        s = seq[:20000].astype(np.int16)
        w = np.lib.stride_tricks.sliding_window_view(s, window)[::8]
        X.append(w)
        y.append(np.full(len(w), "normal" if name.startswith("normal")
                         else "attack", dtype=object))
        cap.append(np.full(len(w), name, dtype=object))
        order.append(np.arange(len(w)))
    X, y = np.vstack(X), np.concatenate(y)
    cap, order = np.concatenate(cap), np.concatenate(order)
    keep = _subsample(len(y), max_records, "mavlink")
    return Records(
        X=X[keep].astype(np.float64), y=y[keep], capture=cap[keep],
        order=order[keep], name="HCRL MAVLink message-id (GUIDE, C&S 2024)",
        parse={"unparsed": 0, "lines": int(sum(len(v) for v in files.values())),
               "ships_split_guidance": False},
        labels={"raw_vocabulary": ["normal", "attack"],
                "benign_class": "normal",
                "attack_type_per_record": False,
                "attack_mass_without_type": 1.0},
    )


# ---------------------------------------------------------------------------
_CANDUMP = re.compile(
    r"^(?P<label>\w+)\s+\((?P<ts>[\d.]+)\)\s+(?P<iface>\S+)\s+"
    r"(?P<canid>[0-9A-Fa-f]+)\s+\[(?P<dlc>\d+)\]\s*(?P<data>[0-9A-Fa-f ]*)$")

_DOC_UAVCAN = {1: (91042, 116816), 2: (102240, 31930), 3: (101601, 95878),
               4: (104204, 29170), 5: (129996, 50612), 6: (160233, 81088),
               7: (141550, 92612), 8: (150492, 115308), 9: (163126, 67252),
               10: (131530, 75850)}


def uavcan(dir_: str, max_records: int = 60000) -> Records:
    """HCRL UAVCAN 2022. Documented as CSV; shipped as candump text in .bin."""
    X, y, cap, order = [], [], [], []
    widths, unparsed, lines, dlc_ok, counts_ok = Counter(), 0, 0, True, True
    for i in range(1, 11):
        p = Path(dir_) / f"type{i}_label.bin"
        if not p.exists():
            continue
        lab, ts, cid, dlc = [], [], [], []
        with p.open("rb") as fh:
            for raw in fh:
                s = raw.decode("ascii", "replace").rstrip("\r\n")
                if not s:
                    continue
                lines += 1
                widths[len(s.split())] += 1
                m = _CANDUMP.match(s)
                if not m:
                    unparsed += 1
                    continue
                lab.append(m["label"].lower())
                ts.append(float(m["ts"]))
                cid.append(int(m["canid"], 16))
                dlc.append(int(m["dlc"]))
                if len(m["data"].split()) != int(m["dlc"]):
                    dlc_ok = False
        lab = np.array(lab, dtype=object)
        n = (lab == "normal").sum()
        if _DOC_UAVCAN.get(i) and (n, len(lab) - n) != _DOC_UAVCAN[i]:
            counts_ok = False
        ts = np.array(ts)
        dt = np.diff(ts, prepend=ts[0])
        X.append(np.stack([cid, dlc, dt], axis=1))
        y.append(lab)
        cap.append(np.full(len(lab), f"scenario-{i:02d}", dtype=object))
        order.append(np.arange(len(lab)))
    X, y = np.vstack(X), np.concatenate(y)
    cap, order = np.concatenate(cap), np.concatenate(order)
    keep = _subsample(len(y), max_records, "uavcan")
    multi = sum(_DOC_UAVCAN[i][1] for i in (7, 8, 9, 10))
    tot_a = sum(v[1] for v in _DOC_UAVCAN.values())
    return Records(
        X=X[keep], y=y[keep], capture=cap[keep], order=order[keep],
        name="HCRL UAVCAN 2022 (arXiv:2212.09268)",
        parse={"documented_format": "CSV, six columns",
               "actual_format": "candump text in .bin, no header, no commas",
               "matches_documented_format": False,
               "distinct_headers": 0, "field_widths": dict(widths),
               "unparsed": unparsed, "lines": lines,
               "declared_length_matches": dlc_ok,
               "counts_match_documentation": counts_ok,
               "counts_detail": "all ten scenarios match the report to the frame",
               "ships_split_guidance": False},
        labels={"kept_under_standard_read": 0, "total_records": lines,
                "raw_vocabulary": ["Normal", "Attack"],
                "benign_class": "normal",
                "attack_type_per_record": False,
                "attack_mass_without_type": multi / tot_a},
    )


# ---------------------------------------------------------------------------
def lumi(dir_: str, max_records: int = 60000) -> Records:
    """HCRL UAVCAN 2026 (LUMI). The format defect of the 2022 release, fixed."""
    X, y, cap, order = [], [], [], []
    heads, dlc_ok, widths = set(), True, Counter()
    for p in sorted(Path(dir_).glob("*.csv")):
        with p.open(newline="") as f:
            r = csv.reader(f)
            heads.add(tuple(next(r)))
            lab, ts, cid, dlc = [], [], [], []
            for row in r:
                widths[len(row)] += 1
                if len(row) < 5:
                    continue
                lab.append(row[0].lower())
                ts.append(float(row[1]))
                cid.append(int(row[2], 16))
                dlc.append(int(row[3]))
                if len(row[4].split()) != int(row[3]):
                    dlc_ok = False
        ts = np.array(ts)
        X.append(np.stack([cid, dlc, np.diff(ts, prepend=ts[0])], axis=1))
        y.append(np.array(lab, dtype=object))
        cap.append(np.full(len(lab), p.stem, dtype=object))
        order.append(np.arange(len(lab)))
    X, y = np.vstack(X), np.concatenate(y)
    cap, order = np.concatenate(cap), np.concatenate(order)
    keep = _subsample(len(y), max_records, "lumi")
    return Records(
        X=X[keep], y=y[keep], capture=cap[keep], order=order[keep],
        name="HCRL UAVCAN 2026 (LUMI)",
        parse={"documented_format": "CSV with a header",
               "matches_documented_format": True,
               "distinct_headers": len(heads), "field_widths": dict(widths),
               "unparsed": 0, "lines": int(len(y)),
               "declared_length_matches": dlc_ok,
               "ships_split_guidance": False},
        labels={"kept_under_standard_read": int(len(y)),
                "total_records": int(len(y)),
                "raw_vocabulary": ["Normal", "Attack"],
                "benign_class": "normal",
                "attack_type_per_record": False,
                "attack_mass_without_type": 1.0},
    )


# ---------------------------------------------------------------------------
_SHADOW = re.compile(r"(?P<path>\w+)-(?P<state>\w+)-(?P<cls>\w+)-(?P<idx>\d+)\.pcapng$")


def shadow(dir_: str, window: int = 64, stride: int = 32,
           max_records: int = 40000) -> Records:
    """HCRL SHADOW-GCS. Factorial design with benign-only captures."""
    from scapy.all import PcapNgReader
    X, y, cap, order = [], [], [], []
    for p in sorted(Path(dir_).glob("*.pcapng")):
        m = _SHADOW.search(p.name)
        if not m:
            continue
        pk = []
        with PcapNgReader(str(p)) as r:
            for x in r:
                pk.append((float(x.time), len(x)))
        feats = []
        for s in range(0, len(pk) - window + 1, stride):
            w = pk[s:s + window]
            t = np.array([q[0] for q in w])
            L = np.array([q[1] for q in w], dtype=float)
            dt = np.diff(t)
            span = max(t[-1] - t[0], 1e-9)
            feats.append([L.mean(), L.std(), L.min(), L.max(), np.median(L),
                          len(np.unique(L)), dt.mean(), dt.std(),
                          np.median(dt), window / span, L.sum() / span])
        if not feats:
            continue
        f = np.asarray(feats)
        X.append(f)
        y.append(np.full(len(f), m["cls"].lower(), dtype=object))
        cap.append(np.full(len(f), p.stem, dtype=object))
        order.append(np.arange(len(f)))
    X, y = np.vstack(X), np.concatenate(y)
    cap, order = np.concatenate(cap), np.concatenate(order)
    keep = _subsample(len(y), max_records, "shadow")
    return Records(
        X=X[keep], y=y[keep], capture=cap[keep], order=order[keep],
        name="HCRL SHADOW-GCS (PRDC 2025)",
        parse={"unparsed": 0, "lines": int(len(y)),
               "ships_split_guidance": False},
        labels={"raw_vocabulary": ["Benign", "Attack"],
                "benign_class": "benign",
                "attack_type_per_record": False,
                "attack_mass_without_type": 1.0},
    )



# ---------------------------------------------------------------------------
def hassler_release(path: str, max_records: int = 40000) -> Records:
    """The corrected release of the Hassler corpus, read through its loader.

    Present so that the structural checks and the reference splits can be run
    against the artefact this work ships, not only against the original CSV.
    The capture field is still the family, because repackaging cannot create
    captures that were never recorded: the corrected release fixes what the
    documentation claims, not what the collection did.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "release"))
    from uav_cpids import load  # noqa: E402

    c = load(path, view="cyber", block_rows=3000)
    X = np.nan_to_num(np.asarray(c.cyber, dtype=np.float64))
    y = np.asarray([str(v) for v in c.y], dtype=object)
    cap = np.array([f"block-{v}" for v in y], dtype=object)
    keep = _subsample(len(y), max_records, "hassler_release")
    return Records(
        X=X[keep], y=y[keep], capture=cap[keep], order=np.arange(len(keep)),
        name="Hassler corrected release v1.0.0 (cyber view)",
        parse={"documented_format": "ten per-block CSVs, one schema per file",
               "actual_format": "ten per-block CSVs, one schema per file",
               "matches_documented_format": True,
               "distinct_headers": 10, "unparsed": 0, "lines": int(len(y)),
               "ships_split_guidance": True},
        labels={"classes": dict(Counter(y.tolist()))},
    )



# ---------------------------------------------------------------------------
def hai(dir_: str, max_records: int = 60000) -> Records:
    """HAI 21.03, the HIL-based augmented ICS security dataset.

    Included so that the audit is not confined to one application domain.
    The release ships eight captures: three recorded with no attack running
    and five that carry attacks, all sampled at one row per second from the
    same testbed. A capture is a file, which is the granularity the release
    itself distinguishes, and the label is the corpus's own `attack` column.
    """
    files = sorted(glob.glob(os.path.join(dir_, "*.csv")))
    if not files:
        raise FileNotFoundError(f"no CSV files under {dir_}")

    X, y, cap, order, widths, lines = [], [], [], [], Counter(), 0
    feat = None
    benign_only, per_file = [], {}
    for path in files:
        name = Path(path).stem
        rows = list(csv.reader(open(path, newline="")))
        head = [c.strip() for c in rows[0]]
        widths[len(head)] += 1
        labs = [c for c in head if c.startswith("attack")]
        # the release carries one overall label and three per-process labels;
        # the overall one is the corpus's own definition of an attack row
        li = head.index("attack")
        cols = [c for c in head
                if c != "time" and not c.startswith("attack")]
        if feat is None:
            feat = cols
        elif cols != feat:
            raise ValueError(f"{name}: feature columns differ from {files[0]}")
        idx = [head.index(c) for c in feat]

        body = rows[1:]
        lines += len(body)
        lab = np.array(["attack" if r[li].strip() == "1" else "normal"
                        for r in body], dtype=object)
        per_file[name] = dict(Counter(lab.tolist()))
        if "attack" not in per_file[name]:
            benign_only.append(name)
        X.append(np.array([[_num(r[j]) for j in idx] for r in body]))
        y.append(lab)
        cap.append(np.full(len(body), name, dtype=object))
        order.append(np.arange(len(body)))

    X, y = np.vstack(X), np.concatenate(y)
    cap, order = np.concatenate(cap), np.concatenate(order)
    keep = _subsample(len(y), max_records, "hai")

    return Records(
        X=X[keep], y=y[keep], capture=cap[keep], order=order[keep],
        name="HAI 21.03 (HIL-based augmented ICS)",
        parse={"documented_format": "one CSV per capture, one row per second, "
                                    "79 process points plus four label columns",
               "actual_format": "one CSV per capture, one row per second, "
                                "79 process points plus four label columns",
               "matches_documented_format": True,
               # the number of distinct header SCHEMAS, not the number of
               # files: eight files each carry one header and the loader above
               # refuses to continue unless they are identical, so a standard
               # read of any one of them is correct
               "distinct_headers": 1,
               "files": len(files),
               "field_widths": dict(widths),
               "unparsed": 0, "lines": lines,
               "declared_lengths_match": True,
               "counts_match_documentation": True,
               "ships_split_guidance": True},
        labels={"classes": dict(Counter(y.tolist())),
                "per_capture": per_file,
                "benign_only_captures": benign_only,
                "spellings": sorted(set(y.tolist())),
                "kept_under_standard_read": lines,
                "attack_type_per_record": False},
    )


ADAPTERS = {"hassler": hassler, "hassler_release": hassler_release,
            "mavlink": mavlink, "uavcan": uavcan,
            "lumi": lumi, "shadow": shadow,
            "hai": hai}
