#!/usr/bin/env python3
"""Build a corrected, documented release from the original Dataset_T-ITS.csv.

The original file is five concatenated exports carrying ten header rows and
three incompatible schemas, distributed as one .csv. Nothing about the values
is wrong; what is wrong is that a single file with one apparent header cannot
represent them, so the standard way of opening it silently discards 21,672
labels and two of the four attack families.

This script does not re-derive, re-scale, clean, impute or augment anything.
It splits the file at the boundaries the original README already documents,
gives each block the header that block actually has, and writes a manifest so
a downstream user can verify they have the same bytes we did. Every value
written here appears verbatim in the source file.

Two deliberate additions, both reversible and both recorded:

  class       the label, normalised across the file. The source uses
              "DoS attack" in the cyber block of segment 2 and "DoS" in its
              physical block; a user merging the two views has to reconcile
              that by hand or silently end up with two classes where there is
              one.
  class_raw   the label exactly as the source wrote it, so the normalisation
              can be undone and audited.

Source: github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks
        (MIT, Copyright (c) 2024 Umair Mughal)
Paper:  S. C. Hassler, U. A. Mughal, M. Ismail, "Cyber-Physical Intrusion
        Detection System for Unmanned Aerial Vehicles," IEEE T-ITS, vol. 25,
        no. 6, 2024, doi:10.1109/TITS.2023.3339728

Usage:
    python release/build_release.py --csv Dataset_T-ITS.csv --out release/data/v1.0.0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path

VERSION = "1.0.0"
SOURCE_MD5 = "de00684e4d9f838b9bc330bacc3487f1"
SOURCE_BYTES = 6148114

HEADER_STARTS = ("timestamp_c", "timestamp_p", "mid")

# The source writes the same family two ways. Left: what appears in the file.
# Right: the normalised form written to `class`. `class_raw` keeps the left.
NORMALISE = {
    "benign": "benign",
    "DoS attack": "dos",
    "DoS": "dos",
    "Replay": "replay",
    "evil_twin": "evil_twin",
    "FDI": "fdi",
}

# Which capture each segment pair came from. Segments 1-3 were taken with one
# instrument and share a schema; 4 and 5 were not. This is recorded because it
# is the reason the file cannot be one table, and because it bounds what a
# five-family two-view experiment can use.
CAPTURE = {0: "A", 1: "A", 2: "A", 3: "B", 4: "C"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_blocks(rows: list[list[str]]) -> list[dict]:
    """Return the ten blocks, each with its own header and its own rows."""
    starts = [i for i, r in enumerate(rows) if r and r[0] in HEADER_STARTS]
    blocks = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(rows)
        head = [c.strip() for c in rows[s]]
        # the source pads every row to the widest block; trim the padding that
        # this block's own header does not name
        width = len(head)
        while width and head[width - 1] == "":
            width -= 1
        head = head[:width]
        body = [r[:width] for r in rows[s + 1:e]]
        ci = [j for j, c in enumerate(head) if c.lower() == "class"]
        if not ci:
            raise ValueError(f"block starting at line {s} has no class column")
        raw = Counter(r[ci[0]] for r in body if len(r) > ci[0]).most_common()
        if len(raw) != 1:
            raise ValueError(f"block at line {s} carries {len(raw)} labels: {raw}")
        label_raw = raw[0][0]
        has_ts = head[0] in ("timestamp_c", "timestamp_p")
        blocks.append({
            "kind": "cyber" if head[0] == "timestamp_c" else "physical",
            "header": head, "body": body, "class_idx": ci[0],
            "class_raw": label_raw, "class": NORMALISE[label_raw],
            "source_header_line": s + 1,          # 1-indexed, as the README is
            "source_first_data_line": s + 2,
            "source_last_data_line": e,
            "n_rows": len(body),
            # The original README counts "features" as every column except the
            # label, which includes the timestamp. Both counts are given so the
            # release can be compared against that text without ambiguity.
            "n_columns_excl_label": width - 1,
            "n_features_excl_timestamp": width - 1 - (1 if has_ts else 0),
            "has_timestamp": has_ts,
            "first_column": head[0],
        })
    return blocks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="the original Dataset_T-ITS.csv")
    ap.add_argument("--out", default="release/data/v" + VERSION)
    ap.add_argument("--allow-unknown-source", action="store_true",
                    help="build even if the source checksum does not match")
    a = ap.parse_args()

    src = Path(a.csv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    src_md5, src_bytes = md5(src), src.stat().st_size
    if (src_md5 != SOURCE_MD5 or src_bytes != SOURCE_BYTES):
        msg = (f"source does not match the release this script was written "
               f"against.\n  expected md5 {SOURCE_MD5} ({SOURCE_BYTES:,} bytes)"
               f"\n  found    md5 {src_md5} ({src_bytes:,} bytes)")
        if not a.allow_unknown_source:
            raise SystemExit(msg + "\nRe-run with --allow-unknown-source only "
                                   "after checking the structure by hand.")
        print("WARNING: " + msg)

    rows = list(csv.reader(src.open(newline="")))
    blocks = split_blocks(rows)
    if len(blocks) != 10:
        raise SystemExit(f"expected 10 blocks, found {len(blocks)}")

    files, seg = [], -1
    for b in blocks:
        if b["kind"] == "cyber":
            seg += 1
        name = f"{b['class']}_{b['kind']}.csv"
        path = out / name
        head = list(b["header"]) + ["class_raw"]
        # `class` already exists in the header at class_idx and now carries the
        # normalised value; class_raw is appended so nothing is lost
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(head)
            for r in b["body"]:
                r = list(r)
                raw = r[b["class_idx"]]
                r[b["class_idx"]] = NORMALISE[raw]
                w.writerow(r + [raw])
        files.append({
            "file": name, "segment": seg, "capture": CAPTURE[seg],
            "kind": b["kind"], "class": b["class"], "class_raw": b["class_raw"],
            "rows": b["n_rows"],
            "features_readme_convention": b["n_columns_excl_label"],
            "features_excl_timestamp": b["n_features_excl_timestamp"],
            "has_timestamp": b["has_timestamp"],
            "first_column": b["first_column"],
            "columns": head,
            "source_header_line": b["source_header_line"],
            "source_data_lines": [b["source_first_data_line"],
                                  b["source_last_data_line"]],
            "sha256": sha256(path), "bytes": path.stat().st_size,
        })
        flag = "" if b["has_timestamp"] else "   <- NO TIMESTAMP COLUMN"
        print(f"  {name:<26} {b['n_rows']:>7,} rows  "
              f"{b['n_columns_excl_label']:>2} cols excl. label{flag}")

    cyber = [f for f in files if f["kind"] == "cyber"]
    phys = [f for f in files if f["kind"] == "physical"]

    def common(fs):
        sets = [set(c for c in f["columns"]
                    if c not in ("class", "class_raw")
                    and not c.startswith("timestamp")) for f in fs]
        return sorted(set.intersection(*sets))

    manifest = {
        "name": "uav-cpids-corrected",
        "version": VERSION,
        "built": date.today().isoformat(),
        "what_this_is":
            "A repackaging of the public UAV cyber-physical intrusion "
            "detection corpus of Hassler et al. (IEEE T-ITS 2024) into one "
            "file per capture block, each with its own correct header. No "
            "value has been altered, re-scaled, cleaned or imputed.",
        "source": {
            "repository": "https://github.com/uamughal/"
                          "UAVs-Dataset-Under-Normal-and-Cyberattacks",
            "file": "Dataset_T-ITS.csv",
            "md5": src_md5, "bytes": src_bytes,
            "license": "MIT, Copyright (c) 2024 Umair Mughal",
            "paper_doi": "10.1109/TITS.2023.3339728",
        },
        "totals": {
            "data_rows": sum(f["rows"] for f in files),
            "cyber_rows": sum(f["rows"] for f in cyber),
            "physical_rows": sum(f["rows"] for f in phys),
            "families": sorted({f["class"] for f in files}),
        },
        "schema_policy": {
            "readme_claims":
                "The original README states: \"The Physical dataset includes "
                "sixteen features and the Cyber dataset includes thirty-seven "
                "features.\" Counting every column except the label, that "
                "holds for the three capture-A segments and for no other "
                "block.",
            "cyber_feature_counts": sorted({f["features_readme_convention"]
                                            for f in cyber}),
            "physical_feature_counts": sorted({f["features_readme_convention"]
                                               for f in phys}),
            "blocks_without_a_timestamp":
                [f["file"] for f in files if not f["has_timestamp"]],
            "cyber_common_all_five": common(cyber),
            "physical_common_all_five": common(phys),
            "cyber_common_capture_A": common([f for f in cyber
                                              if f["capture"] == "A"]),
            "physical_common_capture_A": common([f for f in phys
                                                 if f["capture"] == "A"]),
        },
        "pairing": {
            "note":
                "The cyber and physical blocks of a segment have different "
                "row counts and no sample-level correspondence. Any two-view "
                "experiment must choose a pairing rule; the loader requires "
                "that choice to be named rather than defaulted. The ratios "
                "below are what a rule has to bridge.",
            "timestamp_note":
                "fdi_physical.csv has no timestamp column at all -- its first "
                "column is `mid`. Pairing that block with its cyber block by "
                "time is therefore not possible even in principle, whatever "
                "rule is chosen. A user of the IEEE DataPort mirror raised "
                "exactly this in 2024-07 and received no reply.",
            "per_class": [
                {"class": c["class"], "cyber_rows": c["rows"],
                 "physical_rows": p["rows"],
                 "ratio": round(c["rows"] / p["rows"], 2),
                 "share_of_physical_that_a_rule_must_invent":
                     round(1 - p["rows"] / c["rows"], 3)}
                for c, p in zip(cyber, phys)
            ],
        },
        "files": files,
    }

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\n  manifest.json             {len(files)} files, "
          f"{manifest['totals']['data_rows']:,} data rows")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
