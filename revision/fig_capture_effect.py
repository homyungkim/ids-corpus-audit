"""Fig. 2 of the revised manuscript: fingerprint index and capture effect
against the smallest number of captures per class. Reads the revision's
measurement files; writes EPS, PDF and a 600-dpi PNG."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

runs, out = Path(sys.argv[1]), Path(sys.argv[2])
SC = json.loads((runs / "split_cost.json").read_text())
FP = json.loads((runs / "fingerprint.json").read_text())
ORDER = ["hassler", "mavlink", "hai", "uavcan", "lumi", "shadow"]
LABEL = {"hassler": "Hassler", "mavlink": "MAVLink", "hai": "HAI 21.03",
         "uavcan": "UAVCAN 2022", "lumi": "LUMI 2026", "shadow": "SHADOW-GCS"}
FPKEY = {"hassler": "hassler_session"}
MODELS = ["lr", "rf", "xgb", "mlp", "cnn"]
# small horizontal offsets so corpora that share kappa stay readable
DX = {"hassler": -0.25, "mavlink": 0.25, "uavcan": -0.3, "lumi": 0.3}
# label offsets in points, per panel, chosen so no two labels overlap
OFF1 = {"hassler": (-44, -3), "mavlink": (6, 2), "hai": (6, -3),
        "uavcan": (-52, 6), "lumi": (6, -9), "shadow": (-30, 8)}
OFF2 = {"hassler": (6, 0), "mavlink": (6, 0), "hai": (6, 0),
        "uavcan": (-8, 44), "lumi": (22, 28), "shadow": (-12, 14)}

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6})
INK, MUTED, ACCENT = "#1f1f1f", "#8a8a8a", "#2b5d8a"

fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.5, 2.6))
for c in ORDER:
    k = SC[c]["kappa"] + DX.get(c, 0)
    v = FP[FPKEY.get(c, c)]
    ib = v["base_rf"]["iota_bal"]
    wc = v["within_class"]["weighted_iota_bal"]
    a1.plot([k, k], [wc, ib], color=MUTED, lw=0.8, zorder=1)
    a1.scatter([k], [ib], s=22, color=ACCENT, zorder=3)
    a1.scatter([k], [wc], s=22, facecolor="white", edgecolor=ACCENT, lw=0.9,
               zorder=3)
    a1.annotate(LABEL[c], (k, ib), xytext=OFF1[c], textcoords="offset points",
                fontsize=7, color=INK)
    P = SC[c]["policies"]
    costs = [P[f"random-row/{m}"]["bal_mean"] - P[f"loco/{m}"]["bal_mean"]
             for m in MODELS]
    a2.plot([k, k], [min(costs), max(costs)], color=MUTED, lw=0.8, zorder=1)
    a2.scatter([k] * len(costs), costs, s=6, color=MUTED, zorder=2)
    a2.scatter([k], [np.mean(costs)], s=26, color=ACCENT, zorder=3)
    a2.annotate(LABEL[c], (k, np.mean(costs)), xytext=OFF2[c],
                textcoords="offset points", fontsize=7, color=INK,
                arrowprops=(dict(arrowstyle="-", color=MUTED, lw=0.5)
                            if c in ("uavcan", "lumi", "shadow") else None))

for ax in (a1, a2):
    ax.set_xticks([2, 5, 10, 14])
    ax.set_xticklabels(["2", "5", "10", "14"])
    ax.minorticks_off()
    ax.set_xlim(-0.6, 16.5)
    ax.set_xlabel("κ, smallest number of captures in any class")
a1.set_ylim(0, 1.02)
a1.set_ylabel("capture-fingerprint index")
a1.set_title("(a) fingerprint: all captures (filled), within class (open)",
             fontsize=7.5, loc="left")
a2.axhline(0, color=MUTED, lw=0.5, ls=":")
a2.set_ylabel("capture effect (balanced accuracy)")
a2.set_title("(b) capture effect: five models (small), mean (large)",
             fontsize=7.5, loc="left")
fig.tight_layout(w_pad=2.0)
out.mkdir(parents=True, exist_ok=True)
for ext, kw in (("eps", {}), ("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(out / f"fig_capture_effect.{ext}", bbox_inches="tight", **kw)
print("written", out)
