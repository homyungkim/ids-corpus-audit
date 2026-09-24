#!/usr/bin/env python3
"""The graphical abstract: what the split policy costs, against captures per class.

Elsevier asks for something legible at 13 x 5 cm, so this is one panel with one
message: the cost of the evaluation protocol is set by how many captures each
class spans, and at one capture the measurement that would settle it cannot be
formed at all. Two series are drawn because the corpus with a 99.2 % majority
class hides most of its cost under plain accuracy.

Values are read from the released measurement files, never typed in.

    python3 scripts/graphical_abstract.py
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
INK, WARN, GREY = "#2166AC", "#D95F02", "#6E6E6E"

# one label per x position; the two UAVCAN releases share x = 10
ANNOT = {"hassler": ("Hassler et al.", (11, -13)),
         "mavlink": ("MAVLink (GUIDE)", (10, 9)),
         "hai": ("HAI 21.03 (ICS)", (-16, -22)),
         "uavcan": ("UAVCAN 22 / 26", (-6, 12)),
         "shadow": ("SHADOW-GCS", (-40, -20))}


def worst_gap(corpus: dict, field: str) -> float:
    """Largest drop in random-forest score from the random-row baseline."""
    base = corpus["policies"]["random-row"]["rf"][field]
    gaps = [v["rf"][field] - base
            for k, v in corpus["policies"].items()
            if k != "random-row" and "defined" not in v]
    return min(gaps) if gaps else float("nan")


def min_captures(entry: dict) -> int:
    for r in entry["results"]:
        if r["code"] == "C1":
            return min(ast.literal_eval(str(r["value"])).values())
    raise KeyError("C1 missing")


def main() -> int:
    sc = sys.argv[1] if len(sys.argv) > 1 else "runs/audit/split_cost.json"
    ck = sys.argv[2] if len(sys.argv) > 2 else "runs/audit/checklist_final.json"
    checks = json.load(open(ck))
    cost = {c["corpus"]: c for c in json.load(open(sc))}
    missing = [k for k in LABEL if k not in cost or k not in checks]
    if missing:
        print("missing:", missing)
        return 1

    pts = sorted(((min_captures(checks[k]),
                   worst_gap(cost[k], "mean"),
                   worst_gap(cost[k], "bal_mean"), k) for k in LABEL))

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7,
                         "savefig.dpi": 600, "figure.facecolor": "white",
                         "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(5.12, 2.05))

    ax.axvspan(0.78, 1.55, color=WARN, alpha=0.09, lw=0)
    ax.text(0.82, -0.055, "capture-out\nundefined here", fontsize=6.3,
            color=WARN, va="center", linespacing=1.3)

    xs = [p[0] for p in pts]
    ax.plot(xs, [p[2] for p in pts], color=INK, lw=1.1, zorder=2,
            marker="o", ms=4.4, mec="white", mew=0.7,
            label="balanced accuracy")
    ax.plot(xs, [p[1] for p in pts], color=GREY, lw=0.9, ls=(0, (4, 3)),
            zorder=1, marker="o", ms=3.4, mfc="white", mec=GREY, mew=0.9,
            label="accuracy")

    for x, acc, bal, k in pts:
        if k not in ANNOT:
            continue
        txt, off = ANNOT[k]
        ax.annotate(txt, (x, bal), textcoords="offset points", xytext=off,
                    fontsize=6.4, color=WARN if x <= 2 else "#1A1A1A")

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"], fontsize=7.2)
    ax.minorticks_off()
    ax.set_xlim(0.78, 30)
    ax.set_ylim(-0.46, 0.09)
    ax.set_yticks([0, -0.1, -0.2, -0.3, -0.4])
    ax.tick_params(labelsize=7.2, length=2.5, width=0.7)
    ax.set_xlabel("smallest number of captures in any class", fontsize=7.4)
    ax.set_ylabel("cost of the\nsplit policy", fontsize=7.4, linespacing=1.3)
    ax.axhline(0, color="#CCCCCC", lw=0.7, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=6.4, frameon=False, loc="lower right",
              handlelength=2.4, borderaxespad=0.1)
    ax.set_title("A record carries a fingerprint of its capture. What that\n"
                 "costs is set by how many captures each class spans.",
                 fontsize=7.6, pad=4, loc="left", color="#1A1A1A",
                 linespacing=1.35)

    os.makedirs("runs/audit", exist_ok=True)
    fig.tight_layout(pad=0.4)
    for ext in ("png", "eps", "pdf", "tif"):
        fig.savefig(f"runs/audit/graphical_abstract.{ext}")
    from PIL import Image
    im = Image.open("runs/audit/graphical_abstract.png")
    w, h = im.size
    print("wrote graphical_abstract.*  %d x %d px  (Elsevier minimum 1328 x 531)"
          % (w, h))
    print("  aspect %.2f (13:5 = 2.60)" % (w / h))
    ok = w >= 1328 and h >= 531
    print("  size requirement:", "met" if ok else "NOT met")
    for x, acc, bal, k in pts:
        print(f"   {LABEL[k]:18s} captures/class {x:2d}"
              f"   accuracy {acc:+.3f}   balanced {bal:+.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
