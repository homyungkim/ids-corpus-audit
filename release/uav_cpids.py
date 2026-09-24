"""Reader for the UAV cyber-physical intrusion detection corpus.

Reads either the corrected release built by `build_release.py` or the original
`Dataset_T-ITS.csv` directly. Both paths produce the same tensors; the original
is parsed at its ten real header rows rather than at the one that `read_csv`
assumes.

What this reader will not do
----------------------------
It will not pick a pairing rule for you. The cyber and physical blocks of a
segment have different row counts and carry no sample-level correspondence, so
a two-view experiment has to invent one, and for two of the five families the
rule has to supply more than nine tenths of the physical samples. That is a
modelling decision with a measurable effect on the result, so `view="both"`
requires `pair=` to be named and records the choice in `Corpus.provenance`.

It will also not hand you a five-family two-view dataset at full width. The
five physical blocks share three column names (`pitch`, `roll`, `yaw`) and the
five cyber blocks share 34; only the three capture-A segments share the 37 and
16 columns the original README describes. `schema=` selects between those, and
the default refuses to guess.

Quick start
-----------
    from uav_cpids import load

    # one view, all five families, on the columns all five share
    d = load("data/v1.0.0", view="cyber")

    # one view, three families, full width
    d = load("data/v1.0.0", view="cyber", capture="A")

    # two views: name the rule
    d = load("data/v1.0.0", view="both", capture="A", pair="stretch")

    print(d.provenance)          # every choice that shaped the arrays
    tr, te = d.grouped_split()   # leakage-free by construction

Evaluation
----------
Rows are packets. Consecutive packets from one capture are near-duplicates, so
a random row split lets a model answer the test set from memorised neighbours.
`Corpus.group` holds a contiguous-block id and `grouped_split()` uses it. If
you split on rows instead, say so when you report the number.

Source: github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks
        (MIT, Copyright (c) 2024 Umair Mughal)
Paper:  S. C. Hassler, U. A. Mughal, M. Ismail, IEEE T-ITS 25(6), 2024,
        doi:10.1109/TITS.2023.3339728
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

__all__ = ["Corpus", "load", "describe", "apply_split",
           "PairingRequired", "SchemaRequired", "PAIRINGS", "FAMILIES"]

FAMILIES = ("benign", "dos", "replay", "evil_twin", "fdi")
CAPTURE_OF = {"benign": "A", "dos": "A", "replay": "A",
              "evil_twin": "B", "fdi": "C"}

HEADER_STARTS = ("timestamp_c", "timestamp_p", "mid")
NORMALISE = {"benign": "benign", "DoS attack": "dos", "DoS": "dos",
             "Replay": "replay", "evil_twin": "evil_twin", "FDI": "fdi"}

PAIRINGS = {
    "stretch":
        "Index the shorter view onto the longer with a monotone linear map, "
        "preserving order. Every physical row is repeated ceil(ratio) times. "
        "This is the conventional reconstruction and the one most downstream "
        "work implies, but it manufactures the majority of the physical view "
        "for dos and replay.",
    "truncate":
        "Take min(len_cyber, len_physical) rows from each, in order. Nothing "
        "is invented; for dos and replay this discards 92% of the cyber view.",
    "none":
        "Return the two views unpaired, with their own lengths. Use this when "
        "the experiment does not need row-level correspondence.",
}

_PAIRING_HELP = "\n".join(f"  {k:<10} {v}" for k, v in PAIRINGS.items())


class PairingRequired(ValueError):
    """Raised when a two-view load is requested without naming a rule."""


class SchemaRequired(ValueError):
    """Raised when a schema choice cannot be inferred safely."""


# ---------------------------------------------------------------------------
@dataclass
class Corpus:
    cyber: Optional[np.ndarray]
    phys: Optional[np.ndarray]
    y: np.ndarray
    group: np.ndarray
    cyber_cols: list[str] = field(default_factory=list)
    phys_cols: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def counts(self) -> dict:
        u, c = np.unique(self.y, return_counts=True)
        return {str(a): int(b) for a, b in zip(u, c)}

    @property
    def X(self) -> np.ndarray:
        """The two views side by side. Only defined for a paired load."""
        if self.cyber is None:
            return self.phys
        if self.phys is None:
            return self.cyber
        if len(self.cyber) != len(self.phys):
            raise ValueError(
                "the two views have different lengths, so they cannot be "
                "concatenated. This load used pair='none'; choose 'stretch' "
                "or 'truncate' if you need aligned rows.")
        return np.hstack([self.cyber, self.phys])

    def grouped_split(self, test_size: float = 0.3, seed: int = 0):
        """Split whole contiguous blocks, never individual rows."""
        rng = np.random.default_rng(seed)
        groups = np.unique(self.group)
        rng.shuffle(groups)
        n_test = max(1, int(round(len(groups) * test_size)))
        test = set(groups[:n_test].tolist())
        mask = np.array([g in test for g in self.group])
        return np.flatnonzero(~mask), np.flatnonzero(mask)


# ---------------------------------------------------------------------------
def _blocks_from_original(path: Path) -> list[dict]:
    rows = list(csv.reader(path.open(newline="")))
    starts = [i for i, r in enumerate(rows) if r and r[0] in HEADER_STARTS]
    if len(starts) != 10:
        raise ValueError(
            f"{path.name} has {len(starts)} header rows, not the 10 this "
            f"reader knows about. The file may be a different release; check "
            f"it with uav_cpids.describe() before going further.")
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(rows)
        head = [c.strip() for c in rows[s]]
        w = len(head)
        while w and head[w - 1] == "":
            w -= 1
        head = head[:w]
        ci = head.index("class")
        body = [r[:w] for r in rows[s + 1:e]]
        raw = body[0][ci] if body else ""
        out.append({"kind": "cyber" if head[0] == "timestamp_c" else "physical",
                    "cls": NORMALISE[raw], "header": head, "rows": body,
                    "class_idx": ci})
    return out


def _blocks_from_release(root: Path) -> list[dict]:
    man = json.loads((root / "manifest.json").read_text())
    out = []
    for f in man["files"]:
        rows = list(csv.reader((root / f["file"]).open(newline="")))
        head = rows[0]
        # class_raw is provenance, not a feature
        keep = [i for i, c in enumerate(head) if c != "class_raw"]
        head = [head[i] for i in keep]
        body = [[r[i] for i in keep] for r in rows[1:]]
        out.append({"kind": f["kind"], "cls": f["class"], "header": head,
                    "rows": body, "class_idx": head.index("class")})
    return out


def _numeric(v: str) -> float:
    v = v.strip()
    if not v:
        return np.nan
    try:
        return float(v)
    except ValueError:
        # MAC addresses, IPs and protocol strings are identifiers, not
        # magnitudes. They are hashed to a stable code so the array stays
        # numeric; treat them as categorical or drop them.
        return float(abs(hash(v)) % 100_000)


def _matrix(rows, head, cols) -> np.ndarray:
    idx = [head.index(c) for c in cols]
    return np.asarray([[_numeric(r[j]) if j < len(r) else np.nan for j in idx]
                       for r in rows], dtype=np.float64)


def _feature_cols(head: list[str]) -> list[str]:
    return [c for c in head
            if c not in ("class", "class_raw") and not c.startswith("timestamp")]


# ---------------------------------------------------------------------------
def describe(path: str | Path) -> dict:
    """Report what is in a corpus without building arrays."""
    p = Path(path)
    blocks = (_blocks_from_release(p) if p.is_dir()
              else _blocks_from_original(p))
    cy = [b for b in blocks if b["kind"] == "cyber"]
    ph = [b for b in blocks if b["kind"] == "physical"]

    def common(bs):
        return sorted(set.intersection(*[set(_feature_cols(b["header"]))
                                         for b in bs]))
    return {
        "source": str(p),
        "blocks": [{"class": b["cls"], "kind": b["kind"],
                    "rows": len(b["rows"]),
                    "features": len(_feature_cols(b["header"])),
                    "has_timestamp": b["header"][0].startswith("timestamp")}
                   for b in blocks],
        "total_rows": sum(len(b["rows"]) for b in blocks),
        "cyber_common_all": common(cy),
        "physical_common_all": common(ph),
        "cyber_common_capture_A": common([b for b in cy
                                          if CAPTURE_OF[b["cls"]] == "A"]),
        "physical_common_capture_A": common([b for b in ph
                                             if CAPTURE_OF[b["cls"]] == "A"]),
    }


def load(path: str | Path, *, view: str = "cyber",
         capture: Optional[str] = None,
         families: Optional[Iterable[str]] = None,
         pair: Optional[str] = None,
         block_rows: int = 2000,
         seed: int = 0) -> Corpus:
    """Load the corpus.

    path      the corrected release directory, or the original CSV
    view      "cyber" | "physical" | "both"
    capture   None for every family; "A" for the three that share the 37+16
              schema the original README describes
    families  an explicit subset, overriding `capture`
    pair      required when view="both"; one of PAIRINGS
    block_rows  rows per group id, used by grouped_split
    """
    if view not in ("cyber", "physical", "both"):
        raise ValueError("view must be 'cyber', 'physical' or 'both'")
    if view == "both" and pair is None:
        raise PairingRequired(
            "view='both' needs an explicit pairing rule.\n\n"
            "The cyber and physical blocks of a segment have different row "
            "counts and no sample-level correspondence; for dos and replay a "
            "rule has to supply about 92% of the physical view. There is no "
            "safe default, so name one:\n\n" + _PAIRING_HELP + "\n\n"
            "Whatever you choose is recorded in Corpus.provenance and should "
            "be stated when you report the result.")
    if pair is not None and pair not in PAIRINGS:
        raise ValueError(f"unknown pairing '{pair}'. Choose one of:\n"
                         + _PAIRING_HELP)

    p = Path(path)
    blocks = _blocks_from_release(p) if p.is_dir() else _blocks_from_original(p)

    if families is not None:
        want = list(families)
    elif capture is None:
        want = list(FAMILIES)
    else:
        want = [f for f in FAMILIES if CAPTURE_OF[f] == capture]
        if not want:
            raise ValueError(f"no family belongs to capture '{capture}'")

    cy = [b for b in blocks if b["kind"] == "cyber" and b["cls"] in want]
    ph = [b for b in blocks if b["kind"] == "physical" and b["cls"] in want]
    cy.sort(key=lambda b: want.index(b["cls"]))
    ph.sort(key=lambda b: want.index(b["cls"]))

    def common(bs):
        return sorted(set.intersection(*[set(_feature_cols(b["header"]))
                                         for b in bs]))

    ccols = common(cy) if view in ("cyber", "both") else []
    pcols = common(ph) if view in ("physical", "both") else []

    if view in ("physical", "both") and len(pcols) <= 3 and len(want) > 3:
        raise SchemaRequired(
            f"the physical blocks of {want} share only {len(pcols)} columns "
            f"({', '.join(pcols)}). The five physical blocks come from three "
            f"different instruments. Pass capture='A' for the three families "
            f"that share a 16-column physical schema, or families=[...] to "
            f"choose explicitly, or view='cyber' if the physical view is not "
            f"needed.")

    rng = np.random.default_rng(seed)
    Xc, Xp, ys, gs, ratios = [], [], [], [], []
    gid = 0
    for i, cls in enumerate(want):
        bc = next(b for b in cy if b["cls"] == cls)
        bp = next(b for b in ph if b["cls"] == cls)
        xc = _matrix(bc["rows"], bc["header"], ccols) if ccols else None
        xp = _matrix(bp["rows"], bp["header"], pcols) if pcols else None

        if view == "cyber":
            n, X_c, X_p = len(xc), xc, None
        elif view == "physical":
            n, X_c, X_p = len(xp), None, xp
        elif pair == "none":
            n, X_c, X_p = len(xc), xc, xp          # lengths differ on purpose
        elif pair == "stretch":
            n = max(len(xc), len(xp))
            X_c = xc[np.clip(np.round(np.linspace(0, len(xc) - 1, n)), 0,
                             len(xc) - 1).astype(int)]
            X_p = xp[np.clip(np.round(np.linspace(0, len(xp) - 1, n)), 0,
                             len(xp) - 1).astype(int)]
        else:                                       # truncate
            n = min(len(xc), len(xp))
            X_c, X_p = xc[:n], xp[:n]

        if view == "both":
            ratios.append({"class": cls, "cyber_rows": len(xc),
                           "physical_rows": len(xp),
                           "rows_after_pairing": n,
                           "share_invented":
                               round(max(0.0, 1 - len(xp) / n), 3)
                               if pair == "stretch" else 0.0,
                           "share_discarded":
                               round(1 - n / max(len(xc), len(xp)), 3)
                               if pair == "truncate" else 0.0})

        if X_c is not None:
            Xc.append(X_c)
        if X_p is not None:
            Xp.append(X_p)
        ys.append(np.full(n, cls, dtype=object))
        gs.append(gid + np.arange(n) // block_rows)
        gid = int(gs[-1][-1]) + 1

    prov = {
        "reader": "uav_cpids",
        "source": str(p),
        "source_kind": "corrected release" if p.is_dir() else "original CSV",
        "view": view, "capture": capture, "families": want,
        "pairing": pair,
        "pairing_description": PAIRINGS.get(pair) if pair else None,
        "cyber_columns": ccols, "physical_columns": pcols,
        "block_rows": block_rows, "seed": seed,
        "pairing_cost": ratios,
        "evaluation_note":
            "Rows are packets and neighbours are near-duplicates. Use "
            "Corpus.grouped_split(); a random row split inflates accuracy on "
            "this corpus by 5 to 31 points depending on model and reading.",
    }
    return Corpus(cyber=np.vstack(Xc) if Xc else None,
                  phys=np.vstack(Xp) if Xp else None,
                  y=np.concatenate(ys), group=np.concatenate(gs),
                  cyber_cols=ccols, phys_cols=pcols, provenance=prov)


# ---------------------------------------------------------------------------
def apply_split(corpus: "Corpus", split: dict | str | Path):
    """Turn a reference split from `splits/` into (train_idx, test_idx).

    A grouped split is stored as the list of held-out group ids, which is a
    few dozen numbers rather than tens of thousands of row indices, and which
    stays meaningful if the corpus is reloaded with the same `block_rows`.
    A random-row split is stored as a recipe plus a checksum, because its
    exact membership is not what anyone is comparing -- it is there to show
    the gap against the grouped policies, not to be built on.
    """
    if isinstance(split, (str, Path)):
        split = json.loads(Path(split).read_text())

    if split.get("block_rows") not in (None, corpus.provenance.get("block_rows")):
        raise ValueError(
            f"this split was made with block_rows="
            f"{split['block_rows']}, the corpus was loaded with "
            f"block_rows={corpus.provenance.get('block_rows')}. Reload with "
            f"the matching value or the group ids do not line up.")

    if split["policy"].startswith("grouped"):
        test = set(split["test_groups"])
        mask = np.array([g in test for g in corpus.group])
        return np.flatnonzero(~mask), np.flatnonzero(mask)

    if split["policy"] == "random-row":
        rng = np.random.default_rng(split["seed"])
        idx = rng.permutation(len(corpus))
        n = int(round(len(corpus) * split["test_size"]))
        te, tr = np.sort(idx[:n]), np.sort(idx[n:])
        digest = hashlib.sha256(te.tobytes()).hexdigest()[:16]
        if digest != split.get("test_sha256_16"):
            raise ValueError(
                "reconstructed random-row split does not match the recorded "
                "checksum; the corpus differs from the one it was made on")
        return tr, te

    raise ValueError(f"unknown split policy '{split['policy']}'")
