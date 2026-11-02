"""Shared helpers for the measurements added in revision (COMNET-D-26-07272).

Everything here reuses the published adapters unchanged, so a record set loaded
for the revision is byte-identical to the one the first submission measured.
What is added:

* a recording-session identifier for the Hassler corpus, inferred from gaps in
  its own timestamps (the release carries no session field);
* a fixed model zoo, with hyperparameters set once and never tuned per split;
* metrics computed from pooled out-of-fold predictions.
"""
from __future__ import annotations

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "audit"))

from adapters import ADAPTERS, _subsample  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)

KAPPA_FILE = {  # minimum captures per class, file/capture level, as published
    "hassler": 1, "mavlink": 2, "hai": 5, "uavcan": 10, "lumi": 10,
    "shadow": 14}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load(key: str, path: str):
    """Load a corpus through its published adapter, with an on-disk cache."""
    f = CACHE / f"{key}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    r = ADAPTERS[key](path)
    if key == "hassler":
        r.views = dict(r.views or {})
        r.views["hassler_timestamps"] = hassler_timestamps(path)
    f.write_bytes(pickle.dumps(r))
    return r


def hassler_timestamps(path: str, cap: int = 40000) -> dict:
    """Cyber-block timestamps of the Hassler corpus, aligned to the adapter.

    The adapter stacks the five cyber blocks in file order and then draws a
    fixed subsample. Re-reading the timestamp column in the same order and
    applying the same draw gives one timestamp per retained record.
    """
    rows = list(csv.reader(Path(path).open(newline="")))
    starts = [i for i, r in enumerate(rows)
              if r and r[0] in ("timestamp_c", "timestamp_p", "mid")]
    ts, blk = [], []
    for k, s in enumerate(starts):
        if rows[s][0] != "timestamp_c":
            continue
        e = starts[k + 1] if k + 1 < len(starts) else len(rows)
        body = rows[s + 1:e]
        ts.append(np.array([float(r[0]) for r in body]))
        blk.append(np.full(len(body), k))
    t, b = np.concatenate(ts), np.concatenate(blk)
    keep = _subsample(len(t), cap, "hassler")
    return {"t_full": t, "block_full": b, "keep": keep}


def hassler_sessions(r, gap_s: float = 60.0, min_rows: int = 50) -> np.ndarray:
    """Recording-session id per retained Hassler record.

    A new session starts wherever consecutive cyber records within a block are
    more than `gap_s` seconds apart (a negative step, as at a day boundary,
    also starts one). Fragments shorter than `min_rows` records are merged
    into the preceding session of the same block, so a stray packet after a
    pause does not become a session of its own. The id is computed on the full
    block and then restricted to the subsample, so the subsample cannot change
    where a boundary falls.
    """
    h = r.views["hassler_timestamps"]
    t, b = h["t_full"], h["block_full"]
    sid = np.empty(len(t), dtype=np.int64)
    cur = 0
    for blk in np.unique(b):
        idx = np.where(b == blk)[0]
        tt = t[idx]
        brk = np.concatenate([[True], (np.diff(tt) > gap_s) | (np.diff(tt) < 0)])
        seg = np.cumsum(brk) - 1
        # merge short fragments into the previous segment of the same block
        sizes = np.bincount(seg)
        remap, last = {}, None
        for s_ in range(len(sizes)):
            if sizes[s_] < min_rows and last is not None:
                remap[s_] = last
            else:
                remap[s_] = s_
                last = s_
        seg = np.array([remap[s_] for s_ in seg])
        _, seg = np.unique(seg, return_inverse=True)
        sid[idx] = cur + seg
        cur += seg.max() + 1
    lab = np.array([f"s{v:03d}" for v in sid], dtype=object)
    return lab[h["keep"]]


def hassler_instrumentation(r) -> np.ndarray:
    """Instrumentation group A/B/C per record (Table 3 of the manuscript)."""
    grp = {"benign": "A", "dos": "A", "replay": "A", "evil_twin": "B",
           "fdi": "C"}
    return np.array([grp[str(v)] for v in r.y], dtype=object)


# ---------------------------------------------------------------------------
# models: fixed hyperparameters, fitted on the training part only
# ---------------------------------------------------------------------------
class CNN1D:
    """A small 1-D convolutional classifier over the feature vector.

    Treating the feature vector as a one-channel sequence is the form used by
    several of the UAV detectors cited in the manuscript. Standardisation is
    fitted on the training part only. Six epochs, Adam, fixed seed.
    """

    def __init__(self, seed: int = 0, epochs: int = 6, batch: int = 1024):
        self.seed, self.epochs, self.batch = seed, epochs, batch

    def fit(self, X, y):
        import torch
        from torch import nn
        torch.manual_seed(self.seed)
        torch.set_num_threads(2)
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-8
        Z = ((X - self.mu) / self.sd).astype(np.float32)
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        d, k = Z.shape[1], len(self.classes_)
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv1d(16, 16, 3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(8), nn.Flatten(),
            nn.Linear(16 * 8, 64), nn.ReLU(), nn.Linear(64, k))
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        Xt = torch.from_numpy(Z).unsqueeze(1)
        yt = torch.from_numpy(yi.astype(np.int64))
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch):
                j = perm[i:i + self.batch]
                opt.zero_grad()
                lossf(self.net(Xt[j]), yt[j]).backward()
                opt.step()
        return self

    def predict(self, X):
        import torch
        Z = ((X - self.mu) / self.sd).astype(np.float32)
        self.net.eval()
        out = []
        with torch.no_grad():
            Xt = torch.from_numpy(Z).unsqueeze(1)
            for i in range(0, len(Xt), 4096):
                out.append(self.net(Xt[i:i + 4096]).argmax(1).numpy())
        return self.classes_[np.concatenate(out)]


def make_model(kind: str, seed: int = 0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "lr":
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000))
    if kind == "rf":
        return RandomForestClassifier(n_estimators=150, n_jobs=2,
                                      random_state=seed)
    if kind == "xgb":
        return _XGB(seed)
    if kind == "mlp":
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=1000, early_stopping=True,
                                           n_iter_no_change=15,
                                           random_state=seed))
    if kind == "cnn":
        return CNN1D(seed)
    raise ValueError(kind)


class _XGB:
    """XGBoost with string labels; fixed settings."""

    def __init__(self, seed):
        self.seed = seed

    def fit(self, X, y):
        from xgboost import XGBClassifier
        self.classes_ = np.unique(y)
        yi = np.searchsorted(self.classes_, y)
        self.m = XGBClassifier(n_estimators=200, max_depth=6,
                               learning_rate=0.1, subsample=0.8,
                               colsample_bytree=0.8, tree_method="hist",
                               n_jobs=2, random_state=self.seed)
        self.m.fit(np.nan_to_num(X), yi)
        return self

    def predict(self, X):
        return self.classes_[self.m.predict(np.nan_to_num(X)).astype(int)]


def fit_predict(kind, X, y, tr, te, seed=0):
    """Fit on tr, predict te. A class missing from tr is reported, not hidden."""
    m = make_model(kind, seed)
    Xtr = np.nan_to_num(X[tr])
    m.fit(Xtr, y[tr])
    return m.predict(np.nan_to_num(X[te]))


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def metrics(y_true, y_pred, classes=None) -> dict:
    from sklearn.metrics import accuracy_score, recall_score
    classes = np.unique(y_true) if classes is None else classes
    rec = recall_score(y_true, y_pred, labels=classes, average=None,
                       zero_division=0)
    present = np.isin(classes, np.unique(y_true))
    return {"acc": float(accuracy_score(y_true, y_pred)),
            # balanced accuracy over the classes present in y_true
            "bal": float(np.mean(rec[present])),
            "recall": {str(c): float(v) for c, v, p in
                       zip(classes, rec, present) if p}}


def spearman_exact(x, y) -> tuple[float, float]:
    """Spearman rho and exact two-sided permutation p-value (n small)."""
    from itertools import permutations
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)

    def rho(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    r0 = rho(rx, ry)
    cnt = tot = 0
    for p in permutations(range(len(ry))):
        tot += 1
        if abs(rho(rx, ry[list(p)])) >= abs(r0) - 1e-12:
            cnt += 1
    return r0, cnt / tot
