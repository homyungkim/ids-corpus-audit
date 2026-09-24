"""A runnable checklist for intrusion-detection corpora.

Every check here exists because one of five public UAV corpora failed it, and
each returns a number rather than an opinion. The point of writing it as code
rather than as a list of questions is that a corpus can be handed to it and
the answer does not depend on who is asking.

The checks are grouped by what they protect:

  A  format      does the file parse the way its documentation says
  B  labels      does a standard read keep the labels
  C  design      can the corpus separate a class from the capture it came from
  D  evaluation  what does a reported number depend on
  E  views       when there are two modalities, do they correspond

Group C is the one that matters most and the one nobody runs. A window of a
capture carries that capture's fingerprint; a corpus with one capture per class
therefore cannot distinguish attack detection from capture identification, and
no amount of modelling care fixes that after the fact. C4 measures the
fingerprint directly.

A corpus is described to the checklist through `Records`. Writing that adapter
is the only work required to audit a new corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("checklist")

PASS, WARN, FAIL, NA = "PASS", "WARN", "FAIL", "n/a"


# ---------------------------------------------------------------------------
@dataclass
class Records:
    """What a corpus has to provide to be checked.

    X        (n, d) numeric per-record or per-window features
    y        (n,)   class label, as the corpus defines it
    capture  (n,)   which capture/session/file the record came from. This is
                    the field the whole of group C turns on; if a corpus does
                    not let you fill it in honestly, that is itself the finding
    order    (n,)   position within the capture, if the records are sequential
    parse    dict   what the reader observed while parsing (group A)
    labels   dict   what the reader observed about labels (group B)
    views    dict   optional second-modality description (group E)
    """
    X: np.ndarray
    y: np.ndarray
    capture: np.ndarray
    order: Optional[np.ndarray] = None
    parse: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    views: dict = field(default_factory=dict)
    name: str = ""

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Result:
    code: str
    title: str
    status: str
    value: object = None
    detail: str = ""

    def line(self) -> str:
        v = "" if self.value is None else f"  {self.value}"
        return f"  [{self.status:<4}] {self.code}  {self.title}{v}"


# ---------------------------------------------------------------------------
# A -- format
# ---------------------------------------------------------------------------
def A1_parses_as_documented(r: Records) -> Result:
    p = r.parse
    if "documented_format" not in p:
        return Result("A1", "parses under its documented format", NA)
    ok = bool(p.get("matches_documented_format", False))
    return Result("A1", "parses under its documented format",
                  PASS if ok else FAIL,
                  p["documented_format"] if ok else
                  f"documented {p['documented_format']!r}, actually "
                  f"{p.get('actual_format', 'something else')!r}",
                  "A reader following the documentation gets the wrong answer "
                  "and no error." if not ok else "")


def A2_one_header(r: Records) -> Result:
    n = r.parse.get("distinct_headers")
    if n is None:
        return Result("A2", "one header per file, consistent across files", NA)
    return Result("A2", "one header per file, consistent across files",
                  PASS if n == 1 else FAIL, f"{n} distinct header rows",
                  "Embedded headers become data rows under a standard read."
                  if n != 1 else "")


def A3_fixed_field_count(r: Records) -> Result:
    w = r.parse.get("field_widths")
    if not w:
        return Result("A3", "field count is the same on every row", NA)
    k = sorted(w)
    return Result("A3", "field count is the same on every row",
                  PASS if len(k) == 1 else WARN,
                  f"{len(k)} distinct widths: {k[0]}..{k[-1]}",
                  "A delimiter-based read produces ragged rows whose columns "
                  "mean different things." if len(k) > 1 else "")


def A4_unparsed(r: Records) -> Result:
    n, tot = r.parse.get("unparsed", 0), r.parse.get("lines", len(r))
    frac = n / tot if tot else 0.0
    return Result("A4", "every line parses", PASS if n == 0 else FAIL,
                  f"{n:,} of {tot:,} unparsed ({100*frac:.2f}%)")


def A5_declared_lengths(r: Records) -> Result:
    v = r.parse.get("declared_length_matches")
    if v is None:
        return Result("A5", "declared lengths match the payload", NA)
    return Result("A5", "declared lengths match the payload",
                  PASS if v else FAIL, str(v))


def A6_counts_match_documentation(r: Records) -> Result:
    v = r.parse.get("counts_match_documentation")
    if v is None:
        return Result("A6", "record counts match the documentation", NA)
    return Result("A6", "record counts match the documentation",
                  PASS if v else FAIL, r.parse.get("counts_detail", str(v)))


# ---------------------------------------------------------------------------
# B -- labels
# ---------------------------------------------------------------------------
def B1_labels_survive_standard_read(r: Records) -> Result:
    k = r.labels.get("kept_under_standard_read")
    t = r.labels.get("total_records", len(r))
    if k is None:
        return Result("B1", "labels survive the standard read", NA)
    frac = k / t if t else 0.0
    st = PASS if frac > 0.999 else (WARN if frac > 0.95 else FAIL)
    return Result("B1", "labels survive the standard read", st,
                  f"{k:,} of {t:,} ({100*frac:.1f}%)")


def B2_label_vocabulary(r: Records) -> Result:
    v = r.labels.get("raw_vocabulary")
    if v is None:
        return Result("B2", "label vocabulary is internally consistent", NA)
    norm = {}
    for s in v:
        norm.setdefault(str(s).strip().lower().replace(" attack", ""), []).append(s)
    dupes = {k: vv for k, vv in norm.items() if len(vv) > 1}
    return Result("B2", "label vocabulary is internally consistent",
                  PASS if not dupes else WARN,
                  f"{len(v)} spellings" + (f", collisions {dupes}" if dupes else ""),
                  "The same class is written two ways; a merge across files "
                  "yields two classes where there is one." if dupes else "")


def B3_documented_classes_recoverable(r: Records) -> Result:
    doc = r.labels.get("documented_classes")
    if doc is None:
        return Result("B3", "every documented class is recoverable", NA)
    got = set(r.labels.get("recoverable_classes", set(np.unique(r.y).tolist())))
    missing = sorted(set(doc) - got)
    return Result("B3", "every documented class is recoverable",
                  PASS if not missing else FAIL,
                  f"{len(got)} of {len(doc)}" + (f", missing {missing}" if missing else ""))


def B4_type_label_per_record(r: Records) -> Result:
    v = r.labels.get("attack_type_per_record")
    if v is None:
        return Result("B4", "attack type is labelled per record", NA)
    if v is True:
        return Result("B4", "attack type is labelled per record", PASS)
    share = r.labels.get("attack_mass_without_type", 0.0)
    return Result("B4", "attack type is labelled per record",
                  FAIL if share > 0 else WARN,
                  f"{100*share:.1f}% of attack records carry no type",
                  "Type lives in the file name; captures that mix types cannot "
                  "be split by type at all." if share > 0 else "")


# ---------------------------------------------------------------------------
# C -- design. The group that decides whether the corpus can answer anything.
# ---------------------------------------------------------------------------
def C1_captures_per_class(r: Records) -> Result:
    per = {str(c): len(np.unique(r.capture[r.y == c]))
           for c in np.unique(r.y)}
    m = min(per.values())
    st = PASS if m >= 3 else (WARN if m == 2 else FAIL)
    return Result("C1", "at least three captures per class", st, per,
                  "With one capture per class, class and capture are the same "
                  "variable." if m < 2 else "")


def C2_benign_only_capture(r: Records) -> Result:
    benign = r.labels.get("benign_class")
    if benign is None:
        return Result("C2", "benign-only captures exist", NA)
    pure = 0
    for c in np.unique(r.capture):
        lab = set(r.y[r.capture == c].tolist())
        if lab == {benign}:
            pure += 1
    st = PASS if pure >= 2 else (WARN if pure == 1 else FAIL)
    return Result("C2", "benign-only captures exist", st,
                  f"{pure} of {len(np.unique(r.capture))} captures",
                  "Every benign record was taken while an attack was running; "
                  "a benign-trained detector is calibrated on that."
                  if pure == 0 else "")


def C3_leave_one_capture_out_possible(r: Records) -> Result:
    per = {c: len(np.unique(r.capture[r.y == c])) for c in np.unique(r.y)}
    ok = all(v >= 2 for v in per.values())
    return Result("C3", "leave-one-capture-out is possible", PASS if ok else FAIL,
                  f"min {min(per.values())} captures in a class",
                  "No held-out capture exists for at least one class, so "
                  "cross-capture generalisation cannot be measured at all."
                  if not ok else "")


def C4_capture_fingerprint(r: Records, seed: int = 0) -> Result:
    """Can a record name its own capture from the corpus's own features?

    This is the measurement the other checks depend on. If a record identifies
    its capture far above chance, then on a corpus that fails C1 the label is
    readable without reference to the attack.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    caps = np.unique(r.capture)
    if len(caps) < 3:
        return Result("C4", "capture is not identifiable from the features",
                      NA, f"only {len(caps)} captures")
    idx = np.arange(len(r))
    keep = np.isin(r.capture, [c for c in caps
                               if (r.capture == c).sum() >= 10])
    idx = idx[keep[idx]]
    tr, te = train_test_split(idx, test_size=0.3, random_state=seed,
                              stratify=r.capture[idx])
    m = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=seed)
    m.fit(r.X[tr], r.capture[tr])
    acc = accuracy_score(r.capture[te], m.predict(r.X[te]))
    k = len(np.unique(r.capture[idx]))
    chance = 1.0 / k
    # A ratio to chance is not comparable across corpora: with few captures it
    # cannot grow, so a perfect fingerprint on 8 captures would score below a
    # weak one on 35. Normalise into [0, 1] -- 0 is chance, 1 is a capture
    # named every time.
    index = (acc - chance) / (1.0 - chance)
    st = PASS if index < 0.2 else (WARN if index < 0.5 else FAIL)
    return Result("C4", "capture is not identifiable from the features", st,
                  f"index {index:.3f}  ({acc:.4f} on {k}-way, "
                  f"chance {chance:.4f})",
                  "A record names its own capture most of the time. Whether "
                  "that matters depends on C1 and C5." if index >= 0.5 else "")


def C5_class_separable_from_capture(r: Records, seeds: int = 5) -> Result:
    """Does the reported number survive holding a whole capture out?

    This is the check the whole audit exists for. It can only be answered on a
    corpus that passes C3; where C3 fails, the answer is not "fine", it is
    "unknowable", and the check says so.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    per = {c: len(np.unique(r.capture[r.y == c])) for c in np.unique(r.y)}
    if min(per.values()) < 2:
        return Result("C5", "class survives holding a capture out", FAIL,
                      "cannot be measured",
                      "At least one class comes from a single capture, so no "
                      "held-out capture of it exists. The corpus cannot tell "
                      "attack detection from capture identification, and no "
                      "modelling choice repairs that.")
    within, loco = [], []
    for seed in range(seeds):
        tr, te = train_test_split(np.arange(len(r)), test_size=0.3,
                                  random_state=seed, stratify=r.y)
        within.append(_fit(r.X, r.y, tr, te, seed))
        tr2, te2 = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                          random_state=seed).split(r.X, r.y,
                                                                   r.capture))
        loco.append(_fit(r.X, r.y, tr2, te2, seed))
    w, l = float(np.mean(within)), float(np.mean(loco))
    sd = float(np.std(loco))
    gap = w - l
    st = PASS if gap < 0.05 else (WARN if gap < 0.15 else FAIL)
    detail = ""
    if sd > 0.05:
        detail = (f"the held-out-capture score varies by {sd:.3f} across "
                  f"{seeds} seeds (min {min(loco):.4f}); with few captures the "
                  f"split composition dominates, and a single seed misleads.")
    return Result("C5", "class survives holding a capture out", st,
                  f"within {w:.4f} vs capture-out {l:.4f} +/- {sd:.3f} "
                  f"(gap {gap:+.4f})", detail)


# ---------------------------------------------------------------------------
# D -- evaluation
# ---------------------------------------------------------------------------
def _fit(X, y, tr, te, seed):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    m = RandomForestClassifier(n_estimators=150, n_jobs=-1, random_state=seed)
    m.fit(X[tr], y[tr])
    return float(accuracy_score(y[te], m.predict(X[te])))


def D1_split_policy_sensitivity(r: Records, seeds: int = 5) -> Result:
    """Random records versus contiguous blocks, on the same records.

    Unlike C5 this does not need several captures: it groups by position, so
    it applies to a single ordered capture and isolates the neighbour effect.
    """
    from sklearn.model_selection import GroupShuffleSplit, train_test_split
    if r.order is None:
        return Result("D1", "result does not depend on the split policy", NA,
                      "records are not ordered")
    # a block is a contiguous run WITHIN a capture, so the grouping is
    # meaningful whether the corpus is one long capture or many short ones
    blk = np.asarray(r.order) // max(1, len(r) // (len(np.unique(r.capture)) * 8))
    block = np.array([f"{c}#{b}" for c, b in zip(r.capture, blk)], dtype=object)
    if len(np.unique(block)) < 4:
        return Result("D1", "result does not depend on the split policy", NA,
                      "too few blocks")
    rand, grouped = [], []
    for seed in range(seeds):
        tr, te = train_test_split(np.arange(len(r)), test_size=0.3,
                                  random_state=seed, stratify=r.y)
        rand.append(_fit(r.X, r.y, tr, te, seed))
        tr2, te2 = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                          random_state=seed).split(r.X, r.y, block))
        grouped.append(_fit(r.X, r.y, tr2, te2, seed))
    a, b = float(np.mean(rand)), float(np.mean(grouped))
    gap = a - b
    st = PASS if gap < 0.02 else (WARN if gap < 0.10 else FAIL)
    return Result("D1", "result does not depend on the split policy", st,
                  f"random {a:.4f} vs contiguous blocks {b:.4f} "
                  f"(gap {gap:+.4f})")


def D2_headroom(r: Records, seeds: int = 5) -> Result:
    """Is there a task here, or does a shallow model already saturate it?

    Measured under the hardest split the corpus supports, so that saturation
    is a property of the task and not of the evaluation.
    """
    from sklearn.model_selection import GroupShuffleSplit
    per = {c: len(np.unique(r.capture[r.y == c])) for c in np.unique(r.y)}
    if min(per.values()) >= 2:
        group = r.capture
    elif r.order is not None:
        blk = np.asarray(r.order) // max(1, len(r) // 12)
        group = np.array([f"{c}#{b}" for c, b in zip(r.capture, blk)],
                         dtype=object)
    else:
        group = None
    accs, majs = [], []
    for seed in range(seeds):
        if group is not None and len(np.unique(group)) >= 4:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                            random_state=seed).split(r.X, r.y, group))
        else:
            n = len(r); cut = int(n * 0.7)
            o = np.argsort(r.order) if r.order is not None else np.arange(n)
            tr, te = o[:cut], o[cut:]
        accs.append(_fit(r.X, r.y, tr, te, seed))
        majs.append(float(max(np.mean(r.y[te] == c) for c in np.unique(r.y))))
    acc, maj = float(np.mean(accs)), float(np.mean(majs))
    st = PASS if acc < 0.98 else (WARN if acc < 0.995 else FAIL)
    return Result("D2", "the task is not already saturated", st,
                  f"shallow model {acc:.4f} +/- {np.std(accs):.3f}, "
                  f"majority {maj:.4f}",
                  "A plain forest on raw fields solves it; a reported "
                  "improvement has nothing to improve." if acc >= 0.995 else "")


def D3_split_guidance(r: Records) -> Result:
    v = r.parse.get("ships_split_guidance")
    if v is None:
        return Result("D3", "the release states how to split", NA)
    return Result("D3", "the release states how to split",
                  PASS if v else WARN, str(v))


# ---------------------------------------------------------------------------
# E -- two views
# ---------------------------------------------------------------------------
def E1_shared_clock(r: Records) -> Result:
    v = r.views.get("shared_clock")
    if v is None:
        return Result("E1", "the two views share a clock", NA)
    return Result("E1", "the two views share a clock", PASS if v else FAIL,
                  r.views.get("clock_detail", str(v)))


def E2_pairing_cost(r: Records) -> Result:
    per = r.views.get("pairing_cost")
    if per is None:
        return Result("E2", "pairing invents little", NA)
    worst = max(p["share_invented"] for p in per)
    st = PASS if worst < 0.05 else (WARN if worst < 0.5 else FAIL)
    return Result("E2", "pairing invents little", st,
                  f"worst class {100*worst:.1f}% invented")


def E3_fusion_null(r: Records) -> Result:
    v = r.views.get("fusion_null")
    if v is None:
        return Result("E3", "the fusion gain fails its null control", NA)
    gain, null = v["gain_paired"], v["gain_null_class"]
    survives = null >= 0.5 * gain
    return Result("E3", "the fusion gain fails its null control",
                  FAIL if survives else PASS,
                  f"gain {gain:+.3f}, null {null:+.3f}",
                  "Giving each class another class's second view reproduces "
                  "the gain; it measures the capture." if survives else "")


# ---------------------------------------------------------------------------
CHECKS: list[Callable[[Records], Result]] = [
    A1_parses_as_documented, A2_one_header, A3_fixed_field_count,
    A4_unparsed, A5_declared_lengths, A6_counts_match_documentation,
    B1_labels_survive_standard_read, B2_label_vocabulary,
    B3_documented_classes_recoverable, B4_type_label_per_record,
    C1_captures_per_class, C2_benign_only_capture,
    C3_leave_one_capture_out_possible, C4_capture_fingerprint,
    C5_class_separable_from_capture,
    D1_split_policy_sensitivity, D2_headroom, D3_split_guidance,
    E1_shared_clock, E2_pairing_cost, E3_fusion_null,
]

GROUP = {"A": "format", "B": "labels", "C": "design",
         "D": "evaluation", "E": "two views"}


def run(r: Records, checks=None) -> list[Result]:
    out = []
    for fn in (checks or CHECKS):
        try:
            out.append(fn(r))
        except Exception as e:                      # a check must never abort a run
            code = fn.__name__.split("_")[0]
            out.append(Result(code, fn.__name__, NA, None,
                              f"check raised {type(e).__name__}: {e}"))
            log.debug("check %s raised", fn.__name__, exc_info=True)
    return out


def report(name: str, results: list[Result]) -> str:
    lines = [f"\n{name}", "=" * max(60, len(name))]
    last = ""
    for res in results:
        g = res.code[0]
        if g != last:
            lines.append(f"\n{g}  {GROUP.get(g, '')}")
            last = g
        lines.append(res.line())
        if res.detail:
            lines.append(f"         -> {res.detail}")
    tally = {s: sum(1 for x in results if x.status == s)
             for s in (PASS, WARN, FAIL, NA)}
    lines.append("")
    lines.append(f"  {tally[PASS]} pass   {tally[WARN]} warn   "
                 f"{tally[FAIL]} fail   {tally[NA]} not applicable")
    return "\n".join(lines)
