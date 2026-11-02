"""Fig. 1 of the revision: the adjacency curve with and without the timestamp.

Panel (a) is the published curve (runs/audit/leakage.json, naive read, all 37
columns as a plain read keeps them). Panel (b) repeats it with timestamp_c
removed, which is the only change; reader, group sizes, models and the six
seeds are those of scripts/audit_leakage.py. The band marks the two accuracies
on this corpus that the manuscript cites (Gharami and Moni, 2025; Attaullah
et al., 2026).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE.parent))
from audit_benchmark import read_naive  # noqa: E402
from audit_leakage import GRANULARITY, run  # noqa: E402

BAND = {"Gharami and Moni (2025)": 0.9805, "Attaullah et al. (2026)": 0.9874}
STYLE = {"RandomForest": ("#2b5d8a", "-", "o", "random forest"),
         "LogReg": ("#b5651d", "--", "s", "logistic regression"),
         "MLP": ("#6a5a8c", "-.", "^", "multilayer perceptron")}


def curves(rows):
    out = {}
    for m in STYLE:
        mu = [np.mean([r["accuracy"] for r in rows if r["model"] == m and r["g"] == g])
              for g in GRANULARITY]
        sd = [np.std([r["accuracy"] for r in rows if r["model"] == m and r["g"] == g])
              for g in GRANULARITY]
        out[m] = (mu, sd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", required=True)
    a = ap.parse_args()
    outp = Path(a.out)
    pub = [r for r in json.loads(Path(a.published).read_text())["runs"]
           if r["reading"] == "naive-3"]
    seeds = sorted({r["seed"] for r in pub})
    if outp.exists():
        rows = json.loads(outp.read_text())["runs"]
    else:
        ds = read_naive(Path(a.csv))
        keep = [i for i, f in enumerate(ds["features"]) if not f.startswith("timestamp")]
        ds = dict(ds, name="naive-3, no timestamp", X=ds["X"][:, keep],
                  features=[ds["features"][i] for i in keep])
        rows = []
        for g in GRANULARITY:
            for s in seeds:
                rows += run(ds, g, s)
            print(g, {m: round(float(np.mean([r["accuracy"] for r in rows
                                              if r["g"] == g and r["model"] == m])), 3)
                      for m in STYLE})
        outp.write_text(json.dumps({"granularity": list(GRANULARITY),
                                    "seeds": seeds, "runs": rows}, indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)
    lo, hi = min(BAND.values()), max(BAND.values())
    for ax, rr, title in ((axes[0], pub, "(a) plain read, all 37 columns"),
                          (axes[1], rows, "(b) the same, timestamp removed")):
        ax.axhspan(lo, hi, color="#bdbdbd", lw=0, zorder=0)
        for m, (mu, sd) in curves(rr).items():
            c, ls, mk, lab = STYLE[m]
            ax.errorbar(GRANULARITY, mu, yerr=sd, color=c, ls=ls, marker=mk,
                        ms=3.5, lw=1.2, capsize=2, elinewidth=0.8, label=lab)
        ax.set_xscale("log")
        ax.set_xticks(list(GRANULARITY))
        ax.set_xticklabels([("1.5k" if g == 1500 else "3k" if g == 3000 else str(g)) for g in GRANULARITY])
        ax.minorticks_off()
        ax.set_xlabel("group size g (consecutive rows kept together)")
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("accuracy, three classes")
    axes[0].set_ylim(0.3, 1.02)
    axes[1].annotate("accuracies reported on this corpus", xy=(150, (lo + hi) / 2),
                     xytext=(150, 0.94), fontsize=6.5, color="#3d3d3d",
                     ha="center", arrowprops=dict(arrowstyle="-", color="#8a8a8a", lw=0.5))
    axes[1].legend(frameon=False, fontsize=6.5, loc="lower left")
    fig.tight_layout(w_pad=1.5)
    fp = Path(a.fig)
    fp.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("eps", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(fp / f"fig_leakage.{ext}", bbox_inches="tight", **kw)
    print("figure written")


if __name__ == "__main__":
    main()
