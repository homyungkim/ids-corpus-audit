#!/usr/bin/env python3
"""Run the corpus checklist over one or more corpora.

    python audit/run_checklist.py \
        --hassler /path/Dataset_T-ITS.csv \
        --mavlink /path/GUIDE/data/raw \
        --uavcan  /path/uavcan_extracted \
        --lumi    "/path/UAVCAN Attack Dataset 2026 (LUMI)" \
        --shadow  /path/SHADOW-GCS

Anything not given is skipped. The summary table at the end is the one the
paper reports; every cell in it comes out of this run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import ADAPTERS                      # noqa: E402
from checklist import CHECKS, FAIL, NA, PASS, WARN, report, run  # noqa: E402

MARK = {PASS: "o", WARN: "!", FAIL: "X", NA: "-"}


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ADAPTERS:
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--out", default="runs/audit/checklist.json")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")

    all_res, order = {}, []
    for key, fn in ADAPTERS.items():
        path = getattr(a, key)
        if not path:
            continue
        logging.info("loading %s from %s", key, path)
        rec = fn(path)
        logging.info("  %s: %d records, %d captures, classes %s",
                     key, len(rec), len(set(rec.capture.tolist())),
                     sorted(set(rec.y.tolist())))
        res = run(rec)
        print(report(f"{rec.name}   [{key}]", res))
        all_res[key] = {"name": rec.name,
                        "records": int(len(rec)),
                        "captures": int(len(set(rec.capture.tolist()))),
                        "results": [{"code": r.code, "title": r.title,
                                     "status": r.status,
                                     "value": str(r.value) if r.value is not None else None,
                                     "detail": r.detail} for r in res]}
        order.append(key)

    if not order:
        print("nothing to check; pass at least one corpus path")
        return 1

    codes = [r["code"] for r in all_res[order[0]]["results"]]
    print("\n\nsummary")
    print("=" * (14 + 8 * len(order)))
    print(f"{'check':<14}" + "".join(f"{k[:7]:>8}" for k in order))
    print("-" * (14 + 8 * len(order)))
    for i, code in enumerate(codes):
        row = "".join(f"{MARK[all_res[k]['results'][i]['status']]:>8}"
                      for k in order)
        print(f"{code:<14}{row}")
    print("-" * (14 + 8 * len(order)))
    for s in (PASS, WARN, FAIL):
        row = "".join(f"{sum(1 for r in all_res[k]['results'] if r['status'] == s):>8}"
                      for k in order)
        print(f"{s.lower():<14}{row}")
    print("=" * (14 + 8 * len(order)))
    print(f"  {MARK[PASS]} pass   {MARK[WARN]} warn   {MARK[FAIL]} fail   "
          f"{MARK[NA]} not applicable")

    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(all_res, indent=2) + "\n")
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
