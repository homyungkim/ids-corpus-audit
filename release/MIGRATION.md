# Moving an existing pipeline onto this release

If you already have code that reads `Dataset_T-ITS.csv`, this page says what
changes and what it costs. Nothing here requires rewriting a model.

---

## Is my pipeline affected?

Run this against whatever your loader returns.

```python
n_rows   = len(df)
n_classes = df["class"].nunique()
has_phys = any(c in df.columns for c in ("height", "pitch", "barometer", "tof"))
print(n_rows, n_classes, has_phys)
```

| what you see | what it means |
|---|---|
| `33102  3  False` | a single-header read. Two of four attack families and every physical feature are absent. |
| `54783  4  False` | a single-header read without `dropna`; 21,672 labels are empty strings. |
| `42258  5  False` | a segment-aware cyber-only read. Correct. |
| anything with `has_phys=True` and 5 classes | check which physical columns — only `pitch`, `roll`, `yaw` exist in all five. |

---

## The swap

**Before**

```python
import pandas as pd
df = pd.read_csv("Dataset_T-ITS.csv").dropna()
X = df.drop(columns=["class"]).values
y = df["class"].values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
```

**After**

```python
from uav_cpids import load
d = load("data/v1.0.0", view="cyber")        # 42,258 rows, five families
X, y = d.X, d.y
tr, te = d.grouped_split(test_size=0.3, seed=0)
Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
```

If you want the two-view setting, the rule has to be named:

```python
d = load("data/v1.0.0", view="both", capture="A", pair="stretch")
print(d.provenance["pairing_cost"])
# [{'class': 'benign', ..., 'share_invented': 0.545},
#  {'class': 'dos',    ..., 'share_invented': 0.917},
#  {'class': 'replay', ..., 'share_invented': 0.919}]
```

`pair="stretch"` is the conventional reconstruction; it is not a
recommendation. For `dos` and `replay` it supplies about 92% of the physical
rows. Report that number with the result, or use `pair="truncate"`, which
invents nothing and keeps 6,236 rows instead of 33,102.

---

## What your numbers will do

Two things change at once, and they pull in the same direction.

**The task gets harder.** Three classes on 37 cyber columns is not the task the
corpus describes. Holding the feature count fixed at 14 so only the class count
moves:

| model | 3 classes | 5 classes | change |
|---|---:|---:|---:|
| LogisticRegression | 0.740 | 0.539 | −20.2 pt |
| MLP | 0.812 | 0.535 | −27.7 pt |
| RandomForest | 0.885 | 0.867 | −1.8 pt |

**The split gets honest.** Rows are packets; neighbours are near-duplicates.

| model | random row split | whole stretches held out |
|---|---:|---:|
| RandomForest | 0.993 | 0.807 – 0.840 |
| MLP | 0.989 | 0.776 – 0.833 |
| LogisticRegression | 0.849 | 0.642 – 0.745 |

A number in the 0.96 – 1.00 range on this corpus is reachable only under a
row-level split. That is not a reason to hide it — it is a reason to label it.
Reporting both, as `grouped-3000` and `random-row`, costs one extra row in a
table and makes the result comparable to everyone else's.

---

## If you are re-running a published result

The reference splits in `splits/` exist so that "we re-ran X under a
leakage-free split" means the same thing for everyone.

```python
from uav_cpids import load, apply_split
import json

sp = json.load(open("splits/cyber-5__grouped-3000__seed0.json"))
d  = load("data/v1.0.0", block_rows=sp["block_rows"], **sp["load_kwargs"])
tr, te = apply_split(d, sp)
```

Five seeds are provided for each reading and policy. Report the mean and the
spread: at `grouped-3000` there are only 13 – 17 groups, so the variance
between seeds is real and worth showing.

---

## A note on the physical view

Before building anything on the two-view setting, read §"Three facts" in the
README. In short: the five physical blocks share three column names; the FDI
physical block has no timestamp at all; and appending the physical view raises
three-class accuracy by about 30 points even when each class is handed a
*different* class's physical block. That last result means the lift measures
which capture a row came from, not a physical signature of the attack. Whatever
you conclude from the two-view setting, check it against a shuffled control
first.
