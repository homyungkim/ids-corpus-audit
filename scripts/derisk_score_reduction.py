#!/usr/bin/env python3
"""Does the score reduction decide how fragile the threshold is?

The detector scores a window by the mean squared reconstruction error over
T x D entries. At T=128 and D=79 that is an average of 10,112 terms, which
suppresses independent quantization error by roughly sqrt(10112) ~ 100. If
that averaging is what makes the INT8 effect small, then a detector that
scores by the worst entry -- a common choice in practice, because a localised
fault should not be diluted by 10,000 healthy entries -- should be far more
sensitive to the same quantization.

This run trains once and then scores the same windows under several
reductions, so the model, the data and the quantization are held fixed and
only the aggregation varies.
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

from clids import hai                                     # noqa: E402
from clids.data import RawStreams, WindowSet, build_windows  # noqa: E402
from clids.evt import fit_tail                            # noqa: E402
from clids.model import CrossLayerDetector, ModelConfig   # noqa: E402
from clids.train import FakeQuantBackend, TrainConfig, train_model  # noqa: E402

log = logging.getLogger("reduction")

REDUCTIONS = ("mean", "top1pct", "max_step", "max_all")


@torch.no_grad()
def elementwise(model: CrossLayerDetector, ws: WindowSet,
                batch: int = 128) -> dict[str, np.ndarray]:
    """Return one score array per reduction, from the same forward pass."""
    model.eval()
    out: dict[str, list] = {r: [] for r in REDUCTIONS}
    for i in range(0, len(ws), batch):
        sl = slice(i, i + batch)
        xc = torch.from_numpy(ws.xc[sl])
        xp = torch.from_numpy(ws.xp[sl])
        mk = torch.from_numpy(ws.mask[sl])
        rec_c, rec_p, _, _, _ = model(xc, xp, mk)
        mc = mk[..., :1]
        mp = mk[..., 1:2]
        se_c = ((rec_c - xc) ** 2) * mc          # (B, T, Dc)
        se_p = ((rec_p - xp) ** 2) * mp          # (B, T, Dp)
        se = torch.cat([se_c, se_p], dim=-1)     # (B, T, D)
        B, T, D = se.shape
        flat = se.reshape(B, T * D)

        out["mean"].append(flat.mean(dim=1).numpy())
        k = max(1, int(round(0.01 * T * D)))
        out["top1pct"].append(flat.topk(k, dim=1).values.mean(dim=1).numpy())
        out["max_step"].append(se.mean(dim=2).max(dim=1).values.numpy())
        out["max_all"].append(flat.max(dim=1).values.numpy())
    return {r: np.concatenate(v).astype(np.float64) for r, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hai-root", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--max-windows", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/derisk/reduction.json")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    t0 = time.time()

    rec = hai.load(a.hai_root, train_files=hai.DEFAULT_TRAIN, test_files=())
    raw = RawStreams(cyber=rec["cyber"], phys=rec["phys"], label=rec["label"],
                     session=rec["session"], cyber_cols=rec["cyber_cols"],
                     phys_cols=rec["phys_cols"], source="hai")
    ws = build_windows(raw, window=128, overlap=0.5)
    rng = np.random.default_rng(a.seed)
    ws = ws.subset(rng.permutation(len(ws))[:a.max_windows])

    n = len(ws)
    tr = ws.subset(np.arange(0, int(0.50 * n)))
    cal = ws.subset(np.arange(int(0.50 * n), int(0.75 * n)))
    hel = ws.subset(np.arange(int(0.75 * n), n))
    for name in ("xc", "xp"):
        A = getattr(tr, name)
        mu = A.reshape(-1, A.shape[-1]).mean(0)
        sd = A.reshape(-1, A.shape[-1]).std(0) + 1e-6
        for part in (tr, cal, hel):
            setattr(part, name, ((getattr(part, name) - mu) / sd).astype(np.float32))
    log.info("train %d  calib %d  held-out %d", len(tr), len(cal), len(hel))

    mcfg = ModelConfig(d_cyber=tr.xc.shape[-1], d_phys=tr.xp.shape[-1],
                       window=128, width=64, conv_layers=3, patch=16,
                       heads=2, attn_layers=1, n_classes=1, dropout=0.05)
    model = CrossLayerDetector(mcfg)

    class _S:
        pass
    sp = _S(); sp.train, sp.calib, sp.test = tr, cal, hel
    sp.withheld, sp.classes = "none", ["benign"]
    train_model(model, sp,
                TrainConfig(epochs=a.epochs, batch_size=128, lr=1e-3,
                            lambda_cls=0.0, lambda_gate=0.02, p_drop=0.10,
                            burst_prob=0.15, patience=99, num_workers=0,
                            amp=False, seed=a.seed),
                device="cpu")
    log.info("trained in %.0f s", time.time() - t0)

    qb = FakeQuantBackend(model, cal, device="cpu", n_bits=8, n_calib=512)
    fp = elementwise(model, cal)
    q = elementwise(qb.model, cal)

    rows = []
    print("\n" + "=" * 86)
    print(f"{'reduction':<10} {'delta/sd':>9} {'xi':>8} {'local scale':>12} "
          f"{'d/scale':>9} {'inflation':>10} {'FAR 1.0% ->':>12}")
    print("-" * 86)
    for r in REDUCTIONS:
        sf, sq = fp[r], q[r]
        if np.allclose(sf, sq):
            log.error("%s: FP32 and INT8 identical; backends shared", r)
            return 2
        d = float((sq - sf).mean())
        t = fit_tail(sf, percentile=95.0)
        tau = t.threshold(0.01)
        ls = t.local_scale(tau)
        inf = t.inflation(tau, d)
        rows.append({"reduction": r, "delta": d, "delta_over_sd": d / sf.std(),
                     "xi": t.xi, "sigma": t.sigma, "tau": tau,
                     "local_scale": ls, "inflation": inf})
        print(f"{r:<10} {d/sf.std():>+9.4f} {t.xi:>+8.4f} {ls:>12.6g} "
              f"{d/ls:>9.4f} {inf:>9.3f}x {100*0.01*inf:>11.3f}%")
    print("=" * 86)

    p = Path(a.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"note": "de-risking; reduced model, CPU, "
                                     "simulated INT8", "rows": rows}, indent=2))
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
