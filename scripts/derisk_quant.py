#!/usr/bin/env python3
"""De-risking run: does INT8 displace the benign score tail enough to matter?

This is NOT a paper experiment. It trains a reduced model on a CPU for a
handful of epochs and simulates INT8 rather than building a TensorRT engine.
Its only job is to answer, before any GPU time is spent, whether the effect
the manuscript is built on exists at 8 bits on real data. If the displacement
here is indistinguishable from zero, the premise fails and the paper needs a
different claim.

Drift is held fixed on purpose. The calibration pool and the held-out benign
pool are drawn at random from the same operating period, so the only thing
separating the reference scores from the deployed scores is the arithmetic.

Usage:
    python scripts/derisk_quant.py --hai-root <dir with train*.csv.gz>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clids import hai                                    # noqa: E402
from clids.data import RawStreams, build_windows         # noqa: E402
from clids.evt import fit_tail, realized_rate, score_displacement  # noqa: E402
from clids.model import CrossLayerDetector, ModelConfig  # noqa: E402
from clids.train import FakeQuantBackend, TorchBackend, TrainConfig, train_model  # noqa: E402

log = logging.getLogger("derisk")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hai-root", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/derisk/report.json")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()

    # ---- corpus -----------------------------------------------------------
    rec = hai.load(a.hai_root, train_files=hai.DEFAULT_TRAIN, test_files=())
    raw = RawStreams(cyber=rec["cyber"], phys=rec["phys"],
                     label=rec["label"], session=rec["session"],
                     cyber_cols=rec["cyber_cols"], phys_cols=rec["phys_cols"],
                     source="hai")
    log.info("rows %d   control %d   process %d",
             len(raw.label), raw.cyber.shape[1], raw.phys.shape[1])

    ws = build_windows(raw, window=128, overlap=0.5)
    log.info("benign windows %d", len(ws))

    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(ws))[:a.max_windows]
    ws = ws.subset(idx)

    # standardise on the training portion only
    n = len(ws)
    n_tr, n_cal = int(0.50 * n), int(0.25 * n)
    tr = ws.subset(np.arange(0, n_tr))
    cal = ws.subset(np.arange(n_tr, n_tr + n_cal))
    hel = ws.subset(np.arange(n_tr + n_cal, n))
    log.info("train %d   calib %d   held-out benign %d", len(tr), len(cal), len(hel))

    for name in ("xc", "xp"):
        A = getattr(tr, name)
        mu = A.reshape(-1, A.shape[-1]).mean(0)
        sd = A.reshape(-1, A.shape[-1]).std(0) + 1e-6
        for part in (tr, cal, hel):
            setattr(part, name, ((getattr(part, name) - mu) / sd).astype(np.float32))

    # ---- model ------------------------------------------------------------
    mcfg = ModelConfig(d_cyber=tr.xc.shape[-1], d_phys=tr.xp.shape[-1],
                       window=128, width=a.width, conv_layers=3, patch=16,
                       heads=2, attn_layers=1, n_classes=1, dropout=0.05)
    model = CrossLayerDetector(mcfg)
    nparam = sum(p.numel() for p in model.parameters())
    log.info("model parameters %s", f"{nparam:,}")

    class _Split:
        pass
    sp = _Split()
    sp.train, sp.calib, sp.test = tr, cal, hel
    sp.withheld, sp.classes = "none", ["benign"]

    tcfg = TrainConfig(epochs=a.epochs, batch_size=128, lr=1e-3,
                       lambda_cls=0.0, lambda_gate=0.02, p_drop=0.10,
                       burst_prob=0.15, patience=99, num_workers=0,
                       amp=False, seed=a.seed)
    train_model(model, sp, tcfg, device="cpu")
    log.info("trained in %.1f s", time.time() - t0)

    # ---- score under both arithmetics -------------------------------------
    fp32 = TorchBackend(model, device="cpu")
    int8 = FakeQuantBackend(model, cal, device="cpu", n_bits=8, n_calib=512)

    s_cal_fp = np.asarray(fp32.score(cal)[0], dtype=np.float64)
    s_cal_q = np.asarray(int8.score(cal)[0], dtype=np.float64)
    s_hel_fp = np.asarray(fp32.score(hel)[0], dtype=np.float64)
    s_hel_q = np.asarray(int8.score(hel)[0], dtype=np.float64)

    # guard against the failure mode where both backends share a module
    if np.allclose(s_cal_fp, s_cal_q):
        log.error("FP32 and INT8 scores are identical -- the backends are "
                  "sharing a module. Result is meaningless.")
        return 2

    disp = score_displacement(s_cal_fp, s_cal_q)

    # ---- the three rows, measured out of sample ---------------------------
    fit_fp = fit_tail(s_cal_fp, percentile=95.0)
    fit_q = fit_tail(s_cal_q, percentile=95.0)

    rows = []
    for alpha in (0.01, 0.005):
        tau_fp = fit_fp.threshold(alpha)
        tau_q = fit_q.threshold(alpha)
        rows.append({
            "alpha": alpha,
            "tau_fp32": tau_fp,
            "tau_int8_refit": tau_q,
            # every rate below is on held-out benign, never in-sample
            "far_fp32_model_fp32_tau": realized_rate(s_hel_fp, tau_fp),
            "far_int8_model_fp32_tau": realized_rate(s_hel_q, tau_fp),
            "far_int8_model_refit_tau": realized_rate(s_hel_q, tau_q),
            "inflation_measured": realized_rate(s_hel_q, tau_fp) / alpha,
            "inflation_predicted_eq12": fit_fp.inflation(tau_fp, disp["mean"]),
        })

    out = {
        "note": "de-risking run: reduced model, CPU, simulated INT8. "
                "Not publishable numbers.",
        "corpus": "HAI 21.03 train1-3 (benign only)",
        "n_train": len(tr), "n_calib": len(cal), "n_heldout": len(hel),
        "model_params": int(nparam), "epochs": a.epochs,
        "displacement": disp,
        "fit_fp32": fit_fp.to_dict(),
        "fit_int8": fit_q.to_dict(),
        "levels": rows,
        "wall_seconds": round(time.time() - t0, 1),
    }
    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 68)
    print(f"benign score displacement (INT8 - FP32)")
    print(f"   mean {disp['mean']:+.6g}   median {disp['median']:+.6g}"
          f"   std {disp.get('std', float('nan')):.6g}")
    print(f"   as a fraction of the FP32 benign std: "
          f"{disp['mean'] / s_cal_fp.std():+.4f}")
    print(f"\nGPD  FP32  xi={fit_fp.xi:+.4f} sigma={fit_fp.sigma:.6g}"
          f"   INT8  xi={fit_q.xi:+.4f} sigma={fit_q.sigma:.6g}")
    print("\nrealized false-alarm rate on held-out benign")
    print(f"{'alpha':>7} {'FP32 model':>12} {'INT8, old tau':>15} "
          f"{'INT8, refit':>13} {'inflation':>10}")
    for r in rows:
        print(f"{r['alpha']:>7.3f} {100*r['far_fp32_model_fp32_tau']:>11.3f}% "
              f"{100*r['far_int8_model_fp32_tau']:>14.3f}% "
              f"{100*r['far_int8_model_refit_tau']:>12.3f}% "
              f"{r['inflation_measured']:>9.2f}x")
    print("=" * 68)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
