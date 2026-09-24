#!/usr/bin/env python3
"""Checks on the split machinery itself.

    python audit/test_reference_splits.py --release release/data/v1.0.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters                                              # noqa: E402
from make_reference_splits import (blocks_of, hold_out,      # noqa: E402
                                   load_digest, loco_defined)

OK, BAD = [], []


def check(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  -- {detail}" if detail else ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release/data/v1.0.0")
    ap.add_argument("--splits", default="release/splits_corpora")
    a = ap.parse_args()

    print("sampling is independent of call order")
    # the defect this replaced: one module-level generator shared by every
    # adapter, so a corpus drew different records depending on which other
    # corpora were loaded first
    first = adapters._subsample(100_000, 40_000, "uavcan")
    adapters._subsample(100_000, 40_000, "hassler")
    adapters._subsample(100_000, 40_000, "lumi")
    again = adapters._subsample(100_000, 40_000, "uavcan")
    check("same key gives the same draw after other loads",
          np.array_equal(first, again))
    check("different keys give different draws",
          not np.array_equal(first, adapters._subsample(100_000, 40_000, "lumi")))
    check("no subsampling below the cap",
          np.array_equal(adapters._subsample(500, 40_000, "x"), np.arange(500)))

    print("\nthe release loads and its splits match it")
    r = adapters.hassler_release(a.release)
    man = json.loads((Path(a.splits) / "hassler_release" / "manifest.json").read_text())
    check("load digest matches the manifest", load_digest(r) == man["load_digest"],
          f"{load_digest(r)} vs {man['load_digest']}")
    check("row count matches", len(r) == man["rows"])

    print("\ngrouping")
    for size in (500, 3000):
        g = blocks_of(r, size)
        sizes = np.unique(g, return_counts=True)[1]
        full = sizes[sizes == size]
        check(f"blocks of {size}: no group spans two captures",
              all(len({str(x).split('#')[0] for x in g[g == u]}) == 1
                  for u in np.unique(g)[:20]))
        check(f"blocks of {size}: most groups are exactly {size} rows",
              len(full) >= len(sizes) - len(np.unique(r.capture)),
              f"{len(full)} full of {len(sizes)}")
        check(f"blocks of {size}: no group exceeds {size}", sizes.max() <= size,
              f"max {sizes.max()}")

    print("\nheld-out selection")
    g = blocks_of(r, 3000)
    h0, h0b, h1 = (hold_out(g, 0.3, 0), hold_out(g, 0.3, 0), hold_out(g, 0.3, 1))
    check("the same seed selects the same groups", h0 == h0b)
    check("different seeds select different groups", h0 != h1)
    check("about 30% of groups are held out",
          0.2 <= len(h0) / len(np.unique(g)) <= 0.45,
          f"{len(h0)} of {len(np.unique(g))}")
    held = set(h0)
    mask = np.array([str(x) in held for x in g])
    check("train and test share no group",
          not (set(g[mask].tolist()) & set(g[~mask].tolist())))

    print("\nleave-one-capture-out is refused where it is undefined")
    ok, why = loco_defined(r)
    check("refused on a corpus with one capture per class", not ok, why)
    check("the refusal is written down, not silent",
          (Path(a.splits) / "hassler_release" / "loco.UNDEFINED.json").exists())
    check("no loco.json is emitted alongside the refusal",
          not (Path(a.splits) / "hassler_release" / "loco.json").exists())

    # a corpus that does spread its classes over captures must be accepted
    import copy
    r2 = copy.copy(r)
    r2.capture = np.array([f"cap{i % 4}" for i in range(len(r))], dtype=object)
    ok2, why2 = loco_defined(r2)
    check("accepted when every class spans several captures", ok2, why2)

    print(f"\n{len(OK)} passed, {len(BAD)} failed")
    for b in BAD:
        print(f"  failed: {b}")
    return 1 if BAD else 0


if __name__ == "__main__":
    raise SystemExit(main())
