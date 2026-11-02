# Measurements added in revision

These scripts produce every number that the revised manuscript adds or changes.
They reuse the published adapters unchanged, so the records they load are the
records the first submission measured. Results are in `runs/revision/`.

The first-submission results were reproduced exactly before any of this was run:
all 126 checklist values (`audit/run_checklist.py`) and the split-cost table
(`audit/measure_split_cost.py`) agree with `runs/audit/` to the last digit.

| Script | Manuscript | What it measures |
|---|---|---|
| `rev_common.py` | Sections 3.4, 3.5 | shared loading, Hassler session inference, the five model families, metrics |
| `rev_split_cost.py` | Table 11, Sections 5.3–5.4 | three split policies × five models × six corpora; pooled out-of-fold balanced accuracy; capture-level bootstrap; HAI with train1+test3 merged |
| `rev_fingerprint.py` | Table 10, Section 5.2 | index under alternative normalisations, four classifiers, within-class control, size thresholds, equal-sized captures, permutation importance |
| `rev_fusion.py` | Table 7 | fusion arms with session-held-out evaluation, plus two label-free pairings |
| `rev_timing.py` | Sections 4.3–4.6, Table S6 | attack timing inside UAVCAN captures; time overlap between captures |
| `rev_ablation.py` | Section 5.5, Table S4 | adapter settings (subsample, window, stride, session gap) |
| `rev_checklist_sessions.py` | Table 9 | the checklist on the Hassler corpus with sessions as captures |
| `rev_hassler_counters.py` | Section 5.3 | the Hassler measurements without frame.number and wlan.seq |
| `rev_benchmark_notime.py` | Table 6 | the first submission's Table 5 readings with the timestamp removed |
| `rev_leakage_notime.py` | Fig. 1 | the adjacency curve with and without the timestamp |
| `fig_capture_effect.py` | Fig. 2 | fingerprint and capture effect against κ |
| `make_session_artifacts.py` | Sections 7.2–7.3 | `release/data/hassler_sessions.csv`, `loco_sessions.json`, `loco_merged.json` |

## Run

With the corpus paths of the main README in a bash array `FLAGS`:

```bash
cd revision
python3 rev_split_cost.py "${FLAGS[@]}" --splits ../release/splits_corpora --out ../runs/revision/split_cost.json
python3 rev_fingerprint.py "${FLAGS[@]}" --out ../runs/revision/fingerprint.json
python3 rev_ablation.py    "${FLAGS[@]}" --out ../runs/revision/ablation.json
python3 rev_timing.py --uavcan ... --lumi ... --hai ... --out ../runs/revision/timing_uavcan_lumi_hai.json
python3 rev_timing.py --shadow ... --out ../runs/revision/timing_shadow.json
python3 rev_fusion.py --hassler /path/Dataset_T-ITS.csv --out ../runs/revision/fusion.json
python3 rev_checklist_sessions.py --hassler /path/Dataset_T-ITS.csv --out ../runs/revision/checklist_hassler_sessions.json
python3 rev_hassler_counters.py --hassler /path/Dataset_T-ITS.csv --splits ../release/splits_corpora --out ../runs/revision/hassler_nocounters.json
python3 rev_benchmark_notime.py --csv /path/Dataset_T-ITS.csv --out ../runs/revision/benchmark_notime.json
python3 rev_leakage_notime.py --csv /path/Dataset_T-ITS.csv --published ../runs/audit/leakage.json --out ../runs/revision/leakage_notime.json --fig ../runs/revision
python3 fig_capture_effect.py ../runs/revision ../runs/revision
```

Extra dependencies beyond the main requirements: `xgboost`, `torch` (CPU is
enough), `scapy`. On two CPU cores the full set takes about three hours, most of
it the convolutional network under leave-one-capture-out.

Loaded corpora are cached in `revision/cache/` (not committed); delete the
cache to force a fresh load.
