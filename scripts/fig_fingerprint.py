#!/usr/bin/env python3
"""Figure 2: the capture-fingerprint index against captures per class.

Both coordinates are read from the released measurement files. The index and
its spread come from fingerprint_variance.json; the smallest number of
captures in any class comes from check C1 in the merged checklist. Nothing is
typed in, so the figure cannot drift from the numbers in the tables.

    python3 scripts/fig_fingerprint.py
"""

from __future__ import annotations

import ast
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"hassler": "Hassler et al.", "mavlink": "MAVLink (GUIDE)",
         "hai": "HAI 21.03 (ICS)", "uavcan": "UAVCAN 2022",
         "lumi": "UAVCAN 2026", "shadow": "SHADOW-GCS"}
BLUE, RED, GREY = "#2166AC", "#B4453C", "#6E6E6E"

# label offsets: (x multiplier, y offset, horizontal alignment)
OFF = {"hassler": (1.13, 0.028, "left"), "mavlink": (1.13, 0.028, "left"),
       "hai": (1.13, 0.028, "left"), "uavcan": (1.13, 0.028, "left"),
       "lumi": (0.88, -0.017, "right"), "shadow": (0.88, 0.028, "right")}


def min_captures(entry: dict) -> int:
    """Smallest number of captures in any class, from the C1 check."""
    for r in entry["results"]:
        if r["code"] == "C1":
            return min(ast.literal_eval(str(r["value"])).values())
    raise KeyError("C1 missing")


def main() -> int:
    ck = sys.argv[1] if len(sys.argv) > 1 else "runs/audit/checklist_final.json"
    fv = sys.argv[2] if len(sys.argv) > 2 else "runs/audit/fingerprint_variance.json"
    checks = json.load(open(ck))
    index = {c["corpus"]: (c["mean"], c["sd"]) for c in json.load(open(fv))}

    pts = []
    for k in LABEL:
        if k not in checks or k not in index:
            print("missing:", k)
            return 1
        pts.append((k, min_captures(checks[k]), *index[k]))

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.7,
                         "savefig.dpi": 600, "figure.facecolor": "white",
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.6, 3.5))

    ax.axvspan(0.8, 1.6, color="#E8D5D2", alpha=.7, lw=0, zorder=0)
    ax.text(1.02, 0.055, "class and capture\nare the same variable",
            fontsize=7.4, color="#8C2F28", va="bottom", zorder=4)
    ax.axhline(0.5, color=GREY, lw=.8, ls=":", zorder=1)
    ax.text(30, 0.515, "flagged", fontsize=7.4, color=GREY, ha="right")

    for k, c, mu, sd in pts:
        bad = c < 2
        ax.errorbar([c], [mu], yerr=[sd], fmt="none", ecolor=GREY,
                    elinewidth=0.8, capsize=2, zorder=2)
        ax.scatter([c], [mu], s=62, color=RED if bad else BLUE, zorder=3,
                   edgecolor="white", linewidth=.9)
        dx, dy, ha = OFF[k]
        ax.annotate(LABEL[k], (c, mu), xytext=(c * dx, mu + dy), fontsize=8,
                    ha=ha, color=RED if bad else "#1A1A1A")

    ax.set_xscale("log")
    ax.set_xlim(0.8, 34)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("smallest number of captures in any class")
    ax.set_ylabel("capture-fingerprint index  $\\iota$")
    ax.grid(axis="y", color="#EAEAEA", lw=.6, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    os.makedirs("runs/audit", exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "eps", "pdf"):
        fig.savefig(f"runs/audit/fig_fingerprint.{ext}",
                    bbox_inches="tight", pad_inches=.05)
    from PIL import Image
    im = Image.open("runs/audit/fig_fingerprint.png")
    print("wrote fig_fingerprint.{png,eps,pdf}  %d x %d px" % im.size)
    for k, c, mu, sd in sorted(pts, key=lambda t: t[1]):
        print(f"   {LABEL[k]:18s} captures/class {c:2d}   index {mu:.3f} +/- {sd:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
