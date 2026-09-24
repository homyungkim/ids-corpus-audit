#!/usr/bin/env python3
"""Emit reference splits for every corpus the checklist can read.

The audit's complaint about these corpora is that none of them says how to
split, so every paper invents a policy and no two numbers are comparable.
A complaint of that kind is cheap unless the thing complained about is also
supplied, so this script supplies it.

Three policies per corpus:

  loco          leave-one-capture-out. The hardest policy the data supports,
                and the only one under which no test record shares a capture
                with a training record. Emitted only when every class appears
                in at least two captures -- where a class occupies a single
                capture, holding that capture out removes the class, and the
                policy is not merely hard but undefined. That refusal is the
                point: it is check C5 expressed as an artefact.
  grouped-<B>   contiguous runs of B records within a capture held out whole.
                Defined for every corpus with ordered records. Use this when
                LOCO is undefined, and say which B was used.
  random-row    a row-level shuffle. This is what published work on these
                corpora has used. Emitted as a recipe and a checksum rather
                than as a membership list, because its purpose is to let the
                gap against the other two be quantified, not to be built on.

A split is stored as the held-out group or capture identifiers, not as row
indices, so the file stays a few dozen numbers and stays meaningful when the
corpus is reloaded. Reloading has to give the same records for that to hold,
so each corpus also gets a manifest carrying a digest of the load; check it
before trusting a split.

Usage mirrors run_checklist.py:

    python audit/make_reference_splits.py \
        --hassler /path/Dataset_T-ITS.csv \
        --uavcan  /path/uavcan_extracted \
        --out release/splits_corpora
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import ADAPTERS          # noqa: E402
from checklist import Records          # noqa: E402

BLOCKS = (500, 3000)
SEEDS = (0, 1, 2, 3, 4)
TEST_SIZE = 0.3


def load_digest(r: Records) -> str:
    """Identify the loaded records well enough to catch a different load.

    Features are rounded before hashing so that a change of BLAS or of numpy
    version does not invalidate every split on the last bit of a float.
    """
    h = hashlib.sha256()
    h.update(np.asarray(r.y).astype("U32").tobytes())
    h.update(np.asarray(r.capture).astype("U64").tobytes())
    h.update(np.round(np.nan_to_num(r.X.astype(np.float64)), 6).tobytes())
    return h.hexdigest()[:16]


def blocks_of(r: Records, size: int) -> np.ndarray:
    """Contiguous runs of `size` records, never spanning two captures.

    The run index is counted from the start of each capture rather than from
    the start of the corpus, so every run holds `size` records except the last
    one in a capture. Counting globally would make the first run of every
    capture after the first a short remainder, and group sizes would then
    depend on how many records the preceding captures happened to contain.
    """
    if r.order is None:
        raise ValueError("records are not ordered")
    order = np.asarray(r.order)
    within = np.empty(len(order), dtype=np.int64)
    for c in np.unique(r.capture):
        m = r.capture == c
        within[m] = np.argsort(np.argsort(order[m]))
    return np.array([f"{c}#{k}" for c, k in zip(r.capture, within // size)],
                    dtype=object)


def hold_out(groups: np.ndarray, test_size: float, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    u = np.unique(groups)
    rng.shuffle(u)
    k = max(1, int(round(len(u) * test_size)))
    return sorted(str(x) for x in u[:k])


def loco_defined(r: Records) -> tuple[bool, str]:
    per = {c: set(np.unique(r.capture[r.y == c]).tolist())
           for c in np.unique(r.y)}
    thin = {str(c): len(v) for c, v in per.items() if len(v) < 2}
    if thin:
        return False, ("a class occupies a single capture: "
                       + ", ".join(f"{k} in {v}" for k, v in thin.items()))
    return True, f"every class spans at least {min(len(v) for v in per.values())} captures"


def loco_folds(r: Records) -> list[dict]:
    """One fold per capture, skipping folds that would empty a class.

    Holding out a capture is only a usable fold if the classes that capture
    carries still appear in what remains; otherwise the fold trains on a
    label set that does not include the label it is asked to predict.
    """
    folds = []
    for cap in sorted(np.unique(r.capture).tolist()):
        rest = r.y[r.capture != cap]
        held = r.y[r.capture == cap]
        missing = sorted(set(np.unique(held).tolist())
                         - set(np.unique(rest).tolist()))
        if missing:
            folds.append({"held_out_capture": str(cap), "usable": False,
                          "reason": "classes " + ", ".join(map(str, missing))
                                    + " appear nowhere else"})
            continue
        folds.append({"held_out_capture": str(cap), "usable": True,
                      "n_test": int((r.capture == cap).sum()),
                      "test_classes": sorted(map(str, np.unique(held).tolist()))})
    return folds


def random_row_digest(n: int, test_size: float, seed: int) -> str:
    rng = np.random.default_rng(seed)
    te = np.sort(rng.permutation(n)[:int(round(n * test_size))])
    return hashlib.sha256(te.tobytes()).hexdigest()[:16]


def emit(key: str, r: Records, out: Path) -> dict:
    d = out / key
    d.mkdir(parents=True, exist_ok=True)
    rec = {"corpus": key, "name": r.name, "rows": int(len(r)),
           "captures": int(len(np.unique(r.capture))),
           "features": int(r.X.shape[1]),
           "classes": {str(c): int((r.y == c).sum())
                       for c in np.unique(r.y)},
           "load_digest": load_digest(r),
           "test_size": TEST_SIZE, "seeds": list(SEEDS), "policies": {}}

    ok, why = loco_defined(r)
    if ok:
        folds = loco_folds(r)
        (d / "loco.json").write_text(json.dumps(
            {"policy": "loco", "corpus": key, "basis": why,
             "folds": folds}, indent=1) + "\n")
        n_ok = sum(f["usable"] for f in folds)
        rec["policies"]["loco"] = {"folds": len(folds), "usable": n_ok,
                                   "basis": why}
    else:
        (d / "loco.UNDEFINED.json").write_text(json.dumps(
            {"policy": "loco", "corpus": key, "defined": False,
             "reason": why,
             "note": "Leave-one-capture-out cannot be formed on this corpus. "
                     "Any reported accuracy therefore rests on a split that "
                     "leaves capture identity available to the model."},
            indent=1) + "\n")
        rec["policies"]["loco"] = {"defined": False, "reason": why}

    if r.order is not None:
        for b in BLOCKS:
            g = blocks_of(r, b)
            n_g = int(len(np.unique(g)))
            if n_g < 4:
                continue
            for s in SEEDS:
                (d / f"grouped-{b}__seed{s}.json").write_text(json.dumps(
                    {"policy": f"grouped-{b}", "corpus": key,
                     "block_rows": b, "test_size": TEST_SIZE, "seed": s,
                     "held_out_groups": hold_out(g, TEST_SIZE, s)},
                    indent=1) + "\n")
            rec["policies"][f"grouped-{b}"] = {"groups": n_g}

    for s in SEEDS:
        (d / f"random-row__seed{s}.json").write_text(json.dumps(
            {"policy": "random-row", "corpus": key, "n_rows": int(len(r)),
             "test_size": TEST_SIZE, "seed": s,
             "recipe": "np.random.default_rng(seed).permutation(n_rows)"
                       "[:round(n_rows*test_size)], sorted",
             "test_sha256_16": random_row_digest(len(r), TEST_SIZE, s),
             "note": "for measuring the gap against the grouped policies, "
                     "not for building on"}, indent=1) + "\n")
    rec["policies"]["random-row"] = {"rows": int(len(r))}

    (d / "manifest.json").write_text(json.dumps(rec, indent=2) + "\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ADAPTERS:
        ap.add_argument(f"--{k}", default=None)
    ap.add_argument("--out", default="release/splits_corpora")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    out = Path(a.out)

    index = {"test_size": TEST_SIZE, "seeds": list(SEEDS),
             "block_sizes": list(BLOCKS), "corpora": []}
    for key, fn in ADAPTERS.items():
        path = getattr(a, key)
        if not path:
            continue
        logging.info("loading %s from %s", key, path)
        rec = emit(key, fn(path), out)
        index["corpora"].append(rec)
        loco = rec["policies"]["loco"]
        state = (f"{loco['usable']}/{loco['folds']} usable folds"
                 if loco.get("folds") else "UNDEFINED")
        logging.info("  %-8s %6d rows  %3d captures  digest %s  loco %s",
                     key, rec["rows"], rec["captures"],
                     rec["load_digest"], state)

    if not index["corpora"]:
        print("nothing to do; pass at least one corpus path")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.json").write_text(json.dumps(index, indent=2) + "\n")

    print("\n" + "=" * 80)
    print(f"{'corpus':<16}{'rows':>8}{'capt':>6}{'digest':>18}  "
          f"{'leave-one-capture-out':<22}")
    print("-" * 80)
    for c in index["corpora"]:
        lo = c["policies"]["loco"]
        s = (f"{lo['usable']} of {lo['folds']} folds" if lo.get("folds")
             else "undefined")
        print(f"{c['corpus']:<16}{c['rows']:>8,}{c['captures']:>6}"
              f"{c['load_digest']:>18}  {s:<22}")
    print("=" * 80)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
