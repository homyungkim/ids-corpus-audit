#!/usr/bin/env python3
"""How much of a reported number is temporal adjacency?

Rows in this corpus are packets, and consecutive packets from one capture are
near-duplicates of each other. Splitting train from test at the row level puts
a packet and its neighbour on opposite sides, so the test set is answerable
from memorised neighbours rather than from anything the model learned. Papers
on this corpus report accuracies between 0.96 and 1.00; this script measures
what those numbers become as the split is made progressively harder.

The knob is the group size g: rows are grouped into contiguous runs of g, and
whole groups are assigned to train or test. g = 1 reproduces a random row
split exactly; g = 3000 keeps roughly three groups per class, so a group is
a substantial stretch of one capture. Everything else -- reading, features,
classes, model, seed set -- is held fixed, so the curve isolates the split.

The reading used is the naive one, because that is the reading the published
results were produced under.

Usage:
    python scripts/audit_leakage.py --csv /path/to/Dataset_T-ITS.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_benchmark import read_correct, read_naive      # noqa: E402

log = logging.getLogger("leakage")

GRANULARITY = (1, 5, 20, 100, 500, 1500, 3000)
# Reported accuracies on this corpus, for the band drawn on the figure.
REPORTED = {"Gharami & Moni 2025": 0.9805, "Attaullah et al. 2026": 0.9874,
            "Khanfora et al. (quoting)": 0.9613, "Panda & Guo 2025": 1.000}

PALETTE = {"RandomForest": "#2166AC", "LogReg": "#D95F02", "MLP": "#7570B3"}
MARKER = {"RandomForest": "o", "LogReg": "s", "MLP": "^"}
DASH = {"RandomForest": "-", "LogReg": "--", "MLP": "-."}


def run(ds: dict, g: int, seed: int) -> list[dict]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = ds["X"], ds["y"]
    # regroup: contiguous runs of g rows, restarting at every class change so a
    # group never straddles two families
    grp = np.empty(len(y), dtype=np.int64)
    gid, start = 0, 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[start]:
            seg = np.arange(i - start)
            grp[start:i] = gid + seg // g
            gid = grp[i - 1] + 1
            start = i
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                    random_state=seed).split(X, y, grp))
    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr, Xte = imp.transform(X[tr]), imp.transform(X[te])
    out = []
    for name, m in {
        "RandomForest": RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                               random_state=seed),
        "LogReg": make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=300)),
        "MLP": make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=100, random_state=seed)),
    }.items():
        m.fit(Xtr, y[tr])
        p = m.predict(Xte)
        out.append({"reading": ds["name"], "g": g, "seed": seed, "model": name,
                    "n_groups": int(len(set(grp.tolist()))),
                    "accuracy": float(accuracy_score(y[te], p)),
                    "macro_f1": float(f1_score(y[te], p, average="macro"))})
    return out


def figure(rows: list[dict], reading: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7,
                         "savefig.dpi": 300, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(6.0, 3.6))

    lo, hi = min(REPORTED.values()), max(REPORTED.values())
    ax.axhspan(lo, hi, color="#C9C9C9", alpha=0.35, lw=0, zorder=0)
    # label the band clear of the curves, at the right edge where they have
    # already fallen away from it
    ax.text(7800, (lo + hi) / 2, "range reported\non this corpus",
            fontsize=7.5, va="center", ha="left", color="#3D3D3D", zorder=1)

    for model in ("RandomForest", "LogReg", "MLP"):
        gs = sorted({r["g"] for r in rows if r["model"] == model})
        mu = [np.mean([r["accuracy"] for r in rows
                       if r["model"] == model and r["g"] == g]) for g in gs]
        sd = [np.std([r["accuracy"] for r in rows
                      if r["model"] == model and r["g"] == g]) for g in gs]
        ax.errorbar(gs, mu, yerr=sd, color=PALETTE[model], lw=2.0,
                    ls=DASH[model], marker=MARKER[model], ms=5.0,
                    capsize=2.5, elinewidth=1.0, label=model, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.8)

    ax.set_xscale("log")
    ax.set_xlabel("group size $g$   (consecutive rows kept on one side of the split)\n"
                  "$g=1$ is a random row split   ·   $g=3000$ holds out whole stretches of a capture",
                  fontsize=8.5, linespacing=1.6)
    ax.set_ylabel("accuracy")
    ax.set_xlim(0.75, 7000)
    ax.set_ylim(0.45, 1.03)
    ax.set_xticks(list(GRANULARITY))
    ax.set_xticklabels([str(g) for g in GRANULARITY], fontsize=8)
    ax.minorticks_off()
    ax.grid(axis="y", color="#E6E6E6", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    # identity carried by the legend; no end labels, which collided
    ax.legend(frameon=False, fontsize=8, loc="lower left",
              handlelength=2.6, borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-dir", default="runs/audit")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    path = Path(a.csv)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ds in (read_naive(path), read_correct(path, with_physical=False)):
        for g in GRANULARITY:
            for seed in range(a.seeds):
                rows += run(ds, g, seed)
            acc = np.mean([r["accuracy"] for r in rows
                           if r["reading"] == ds["name"] and r["g"] == g])
            log.info("%-12s g=%-5d groups=%-5d mean acc %.4f", ds["name"], g,
                     rows[-1]["n_groups"], acc)

    for reading, stem in (("naive-3", "fig_leakage_naive"),
                          ("correct-5", "fig_leakage_correct")):
        sub = [r for r in rows if r["reading"] == reading]
        if sub:
            figure(sub, reading, out / f"{stem}.png")

    print("\n" + "=" * 72)
    print(f"{'reading':<12}{'g':>6}{'groups':>8}   "
          f"{'RandomForest':>13}{'LogReg':>10}{'MLP':>10}")
    print("-" * 72)
    for reading in ("naive-3", "correct-5"):
        for g in GRANULARITY:
            sel = [r for r in rows if r["reading"] == reading and r["g"] == g]
            if not sel:
                continue
            ng = sel[0]["n_groups"]
            vals = {m: np.mean([r["accuracy"] for r in sel if r["model"] == m])
                    for m in ("RandomForest", "LogReg", "MLP")}
            print(f"{reading:<12}{g:>6}{ng:>8}   "
                  f"{vals['RandomForest']:>13.4f}{vals['LogReg']:>10.4f}"
                  f"{vals['MLP']:>10.4f}")
        print()
    print("=" * 72)

    (out / "leakage.json").write_text(json.dumps(
        {"granularity": list(GRANULARITY), "reported": REPORTED,
         "runs": rows}, indent=2))
    print(f"wrote {out}/leakage.json and the figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
