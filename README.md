# ids-corpus-audit

A runnable audit for intrusion detection corpora, and the artefacts produced by
running it on six public ones.

The tool asks twenty-one questions of a corpus — about its format, its labels,
its design, the evaluation it supports, and the views it carries — and answers
them by parsing the files rather than by reading the documentation. Every table
and figure in the accompanying paper comes out of a single execution of it.

> **Capture Identity as a Confounder in Intrusion Detection Benchmarks:
> A Structural Audit of Six Public Corpora**
> Ho Myung Kim, Department of AI Software, Kyungwoon University.
> Under review. DOI to follow.

---

## What it found

A record carries a fingerprint of the capture session it came from. Where a
corpus records one class per capture, reading that fingerprint and reading the
label are the same operation, and no split, model or regulariser separates
them.

The index below is normalised so corpora with different capture counts are
comparable: 0 is chance, 1 names every capture.

| Corpus | Captures | Min captures per class | Fingerprint index | Leave-one-capture-out |
|---|---:|---:|---|---|
| Hassler et al. (T-ITS 2024) | 5 | 1 | 0.918 ± 0.002 | **undefined** |
| MAVLink (GUIDE) | 8 | 2 | 0.712 ± 0.006 | 8 folds, unstable |
| HAI 21.03 (ICS) | 8 | 5 | 0.959 ± 0.001 | 8 folds |
| UAVCAN 2022 | 10 | 10 | 0.134 ± 0.003 | 10 folds |
| UAVCAN 2026 (LUMI) | 10 | 10 | 0.093 ± 0.004 | 10 folds |
| SHADOW-GCS | 35 | 14 | 0.588 ± 0.005 | 35 folds |

A high index is not by itself a defect. It becomes one only when a class
occupies a single capture. HAI makes that clearest: its index is the highest in
the set, and holding out a whole capture still costs it almost nothing, because
every class spans five captures.

---

## Install

No GPU. No accelerator. Three dependencies.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.11. The full reproduction takes about forty-five minutes on
two CPU cores and peaks under 1 GB of memory.

## Run

`DATASETS.md` says where to obtain each corpus. Anything you do not pass is
skipped, so a partial set still produces its own columns.

```bash
python3 audit/run_checklist.py \
    --hassler /path/Dataset_T-ITS.csv \
    --mavlink /path/GUIDE/data/raw \
    --uavcan  /path/uavcan_extracted \
    --lumi    "/path/UAVCAN Attack Dataset 2026 (LUMI)" \
    --shadow  /path/SHADOW-GCS \
    --hai     /path/hai-21.03 \
    --out runs/audit/checklist.json
```

Then the three measurements the paper reports:

```bash
python3 audit/make_reference_splits.py  ...same corpus flags...  --out release/splits_corpora
python3 audit/measure_split_cost.py     ...same corpus flags...  --out runs/audit/split_cost.json
python3 audit/fingerprint_variance.py   ...same corpus flags...  --out runs/audit/fingerprint_variance.json
```

## Auditing a corpus of your own

Write an adapter. It is the only work required, and it is also where an audit
can go wrong. An adapter fills in five things — features, labels, capture
identifiers, position within the capture, and what the reader observed while
parsing — and the hardest of those to fill in honestly is the capture
identifier. Where a corpus does not distinguish captures, the adapter must say
so rather than inventing a finer granularity the data does not support; an
adapter that invents one makes the design checks pass spuriously.

See `audit/adapters.py`. Each of the six is under sixty lines.

---

## Layout

```
audit/
  checklist.py              the twenty-one checks
  adapters.py               six corpora, one function each
  run_checklist.py          runner; prints the paper's summary table
  make_reference_splits.py  three split policies per corpus
  measure_split_cost.py     what each policy costs, accuracy and balanced
  fingerprint_variance.py   the index with an uncertainty, not a point estimate
  test_reference_splits.py  nineteen checks on the split machinery itself

release/                    a corrected, byte-faithful repackaging of one corpus
  data/v1.0.0/              ten per-block files, each with the header it has
  uav_cpids.py              a loader that refuses to guess
  build_release.py          rebuilds it; refuses unless the source checksum matches
  test_loader.py            verifies the repackaging against the source
  splits/                   reference splits for that release
  splits_corpora/           reference splits for every corpus audited

scripts/                    per-corpus investigations and the figure scripts
runs/audit/                 the JSON and image outputs the paper cites
  checklist.json            the twenty-one checks on all six corpora
  split_cost.json           four split policies, accuracy and balanced
  fingerprint_variance.json the index with its spread over twenty-four runs
  fig_fingerprint.*         Figure 2, and graphical_abstract.* alongside it
```

## Two things the tool does on purpose

**It refuses to produce a split it cannot justify.** Where a class occupies a
single capture, leave-one-capture-out is not merely hard, it is undefined:
holding that capture out removes the class. In that case no split file is
written. In its place the generator emits a record stating that the policy is
undefined and which classes are responsible. A tool that quietly substituted a
weaker policy would conceal the defect this work reports.

**It refuses to evaluate a split against the wrong records.** Every split
directory carries a manifest with a digest of the load it was built from, and
`measure_split_cost.py` stops if the corpus it was handed does not match.

## Reproducibility

Sampling is keyed by corpus name, so a corpus draws the same records whether it
is loaded alone or after five others. An earlier version of `adapters.py` shared
one generator across adapters, which made the draw depend on which corpora
happened to be requested in the same run; `audit/test_reference_splits.py` now
holds that property in place.

Where a corpus is subsampled, figures carry an interval over eight draws and
three evaluation splits each. Where it is not, the budget goes to twenty-four
splits instead. The index should be read to two decimal places; the third is
not resolved by the measurement.

## Licence

MIT, except `release/data/v1.0.0/`, which redistributes MIT-licensed material
from another group under its original notice. See `NOTICE.md`.

## Citing

See `CITATION.cff`, or cite the paper above.
