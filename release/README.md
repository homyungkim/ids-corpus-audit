# uav-cpids-corrected

A corrected, documented repackaging of the public UAV cyber-physical intrusion
detection corpus released with Hassler, Mughal and Ismail, *IEEE Transactions
on Intelligent Transportation Systems* 25(6), 2024.

**No value in this release has been altered, re-scaled, cleaned, imputed or
augmented.** All 54,774 data rows were compared cell by cell against the source
file; there are zero differences. What changed is the packaging.

---

## Why this exists

The source distributes the corpus as one file, `Dataset_T-ITS.csv`. It is not
one table. It is five separate exports concatenated together, carrying **ten**
header rows, and the five do not share a schema.

The original README documents the row ranges accurately. It also states:

> *"The Physical dataset includes sixteen features and the Cyber dataset
> includes thirty-seven features."*

That sentence holds for three of the ten blocks. Evil-twin and FDI were
captured with different instruments and expose different fields in both views.

A reader who trusts that sentence has no reason to suspect heterogeneity, and
so opens the file the way any CSV is opened:

```python
df = pd.read_csv("Dataset_T-ITS.csv")
df = df.dropna()
```

Because the column `pandas` names `class` is index 37 — a position that holds
the label in only three of the ten blocks — this returns:

| | |
|---|---|
| rows with a usable label | **33,102** of 54,774 |
| rows whose label is silently emptied | **21,672 (39.6%)** |
| attack families that survive | **2** of the 4 the paper's abstract names |
| physical features present | **0** |

`evil_twin` (11,156 rows) and `fdi` (4,280 rows) lose **100%** of their labels.

This is a packaging problem, not a data problem, and not a failure of the
people who hit it. This release fixes the packaging.

---

## What is in it

Ten files, one per capture block, each with the header that block actually has.

| file | rows | cols excl. label | capture |
|---|---:|---:|:--:|
| `benign_cyber.csv` | 9,425 | 37 | A |
| `benign_physical.csv` | 4,290 | 16 | A |
| `dos_cyber.csv` | 11,671 | 37 | A |
| `dos_physical.csv` | 973 | 16 | A |
| `replay_cyber.csv` | 12,006 | 37 | A |
| `replay_physical.csv` | 973 | 16 | A |
| `evil_twin_cyber.csv` | 5,683 | 34 | B |
| `evil_twin_physical.csv` | 5,473 | 21 | B |
| `fdi_cyber.csv` | 3,473 | 34 | C |
| `fdi_physical.csv` | 807 | 31 | C |

Plus `manifest.json`: per-file SHA-256, row counts, column lists, the source
line ranges each block came from, and the schema and pairing facts below.

Two columns were added, both reversible:

* **`class`** — the label, normalised. The source writes `DoS attack` in one
  block and `DoS` in its partner, so a user merging the two views ends up with
  two classes where there is one.
* **`class_raw`** — the label exactly as the source wrote it, so the
  normalisation can be undone and audited.

---

## Three facts that bound what you can do with it

**1. The five physical blocks share three column names.**
`pitch`, `roll`, `yaw` — nothing else. Capture A uses a 16-column Tello schema;
capture B a different 21-column SDK schema; capture C 31 fields including
estimator and control signals. Capture C also exposes `mpitch/mroll/myaw`
*alongside* `pitch/roll/yaw`, so even those three may not mean the same thing
across captures.

*Consequence:* a five-family two-view experiment is capped at three physical
features. A full-width two-view experiment exists only within capture A, and
covers three families.

**2. `fdi_physical.csv` has no timestamp column at all.** Its first column is
`mid`. Pairing that block with its cyber block by time is not possible even in
principle, under any rule. A user of the IEEE DataPort mirror raised this in
July 2024 and received no reply.

**3. The two views of a segment have no sample-level correspondence.**

| class | cyber rows | physical rows | ratio | a rule must invent |
|---|---:|---:|---:|---:|
| benign | 9,425 | 4,290 | 2.20× | 54.5% |
| dos | 11,671 | 973 | 11.99× | **91.7%** |
| replay | 12,006 | 973 | 12.34× | **91.9%** |
| evil_twin | 5,683 | 5,473 | 1.04× | 3.7% |
| fdi | 3,473 | 807 | 4.30× | 76.8% |

Any two-view experiment has to choose a pairing rule, and for two families that
rule supplies more than nine tenths of the physical samples. The reader below
refuses to choose for you.

---

## Reading it

```python
from uav_cpids import load

# one view, all five families, on the columns all five share
d = load("data/v1.0.0", view="cyber")

# one view, three families, full width
d = load("data/v1.0.0", view="cyber", capture="A")

# two views: the rule has to be named
d = load("data/v1.0.0", view="both", capture="A", pair="stretch")

print(d.provenance)   # every choice that shaped the arrays, including
                      # how much of the physical view the rule invented
```

`load(..., view="both")` without `pair=` raises `PairingRequired` and prints
the options. Asking for a five-family physical view raises `SchemaRequired`
and explains why. Neither has a default, because there is no safe one.

The reader also accepts the **original** `Dataset_T-ITS.csv` directly and
parses it at its ten real header rows, so you do not need this release to
read the source correctly — you need it to not have to.

---

## Evaluating on it

Rows are packets. Consecutive packets from one capture are near-duplicates, so
a random row split lets a model answer the test set from memorised neighbours.
Measured on this corpus, moving from a random row split to holding out whole
stretches of a capture costs:

| model | random row split | whole stretches held out |
|---|---:|---:|
| RandomForest | 0.993 | 0.807 – 0.840 |
| MLP | 0.989 | 0.776 – 0.833 |
| LogisticRegression | 0.849 | 0.642 – 0.745 |

Published accuracies on this corpus fall between 0.96 and 1.00, which is the
range reachable only when neighbouring packets are split across train and test.

`Corpus.group` carries a contiguous-block id and `Corpus.grouped_split()` uses
it. If you split on rows instead, say so when you report the number.

### Reference splits

Two sets ship with this release.

`splits/` holds splits for the four readings of the corrected release
(cyber-only, capture-A only, and the two pairings), under three policies and
five seeds. These are the splits to use when comparing against the numbers in
the audit paper.

`splits_corpora/hassler/` and `splits_corpora/hassler_release/` hold splits
built by `audit/make_reference_splits.py`, which applies the same three
policies to any corpus the checklist can read. A split is stored as the list
of held-out group identifiers rather than as row indices, so the file stays
small and survives a reload, and each directory carries a `manifest.json`
whose `load_digest` identifies the records the split was built from. Check
that digest before trusting a split; `measure_split_cost.py` refuses to run
if it does not match.

The third policy, leave-one-capture-out, is **not** present for this corpus.
In its place is `loco.UNDEFINED.json`, which records why: every class here
occupies exactly one capture, so holding a capture out removes a class
entirely. That file is check C5 written down as an artefact rather than as a
sentence — the policy that would settle whether a model reads the attack or
the capture cannot be formed on this data at all.

Measured with `audit/measure_split_cost.py`, five seeds, mean ± sd:

| reading | policy | RandomForest | MLP |
|---|---|---:|---:|
| original CSV | random row | 0.9344 ± 0.002 | 0.7911 ± 0.014 |
| original CSV | 500-row stretches | 0.6942 ± 0.077 | 0.6979 ± 0.070 |
| original CSV | 3000-row stretches | 0.5157 ± 0.071 | 0.5301 ± 0.090 |
| corrected release | random row | 0.9333 ± 0.002 | 0.7930 ± 0.006 |
| corrected release | 500-row stretches | 0.6972 ± 0.077 | 0.6870 ± 0.070 |
| corrected release | 3000-row stretches | 0.5170 ± 0.071 | 0.5212 ± 0.071 |

The two readings agree to within 0.004 on every cell. Repackaging the corpus
changed what its documentation claims, not how hard the task is, which is the
intended outcome: a corrected release that moved the numbers would be a
different dataset rather than a better-described one.

Accuracy falls monotonically as the held-out unit grows, from 0.93 to 0.52.
Any number in that range can be obtained on this corpus by choosing a split
policy, so the policy has to be stated with the number.

---

## Reproducing this release

```bash
# 1. get the source (MIT, Copyright (c) 2024 Umair Mughal)
git clone https://github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks

# 2. rebuild -- refuses to run unless the source checksum matches
python build_release.py --csv .../Dataset_T-ITS.csv --out data/v1.0.0

# 3. verify
python test_loader.py --release data/v1.0.0 --original .../Dataset_T-ITS.csv
```

Source checked against: MD5 `de00684e4d9f838b9bc330bacc3487f1`, 6,148,114 bytes.

---

## Licence and attribution

The data is redistributed under the **MIT Licence** of the source repository,
Copyright © 2024 Umair Mughal. The scripts and documentation added here are
released under the same terms.

Cite the corpus:

```bibtex
@article{hassler2024cyber,
  title   = {Cyber-Physical Intrusion Detection System for Unmanned Aerial Vehicles},
  author  = {Hassler, Samuel Chase and Mughal, Umair Ahmad and Ismail, Muhammad},
  journal = {IEEE Transactions on Intelligent Transportation Systems},
  volume  = {25}, number = {6}, year = {2024},
  doi     = {10.1109/TITS.2023.3339728}
}
```

The corpus was collected with support from the US National Science Foundation
EPCN programme under Award 2220346.
