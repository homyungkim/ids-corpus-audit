# Where each corpus comes from, and what is actually in it

Six corpora are audited. Every path below was checked against the live source on
the date of the run recorded in `runs/audit/`. Nothing here is required to read
the results — `runs/audit/` already holds them — but everything here is required
to reproduce them.

The audit tool takes one flag per corpus and skips anything you do not pass, so a
partial set still produces its own columns of the paper's tables.

| # | Corpus | Flag | Account needed |
|---|---|---|---|
| 1 | Hassler et al., cyber-physical UAV | `--hassler` | no |
| 2 | MAVLink message identifiers (GUIDE) | `--mavlink` | no |
| 3 | UAVCAN attack dataset 2022 | `--uavcan` | free HCRL registration |
| 4 | UAVCAN attack dataset 2026 (LUMI) | `--lumi` | free HCRL registration |
| 5 | SHADOW-GCS | `--shadow` | free HCRL registration |
| 6 | HAI 21.03 (industrial control) | `--hai` | no |

---

## 1. Hassler et al. — cyber-physical UAV corpus

The only public UAV corpus with both a network view and a flight view from one
testbed, and the corpus the paper's documentary finding concerns.

```bash
mkdir -p data/hassler
curl -L -o data/hassler/Dataset_T-ITS.csv \
  https://raw.githubusercontent.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks/main/Dataset_T-ITS.csv
# 6,148,114 bytes, MD5 de00684e4d9f838b9bc330bacc3487f1
```

Repository: <https://github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks>
(MIT licence). An IEEE DataPort mirror exists under doi `10.21227/6f22-py65` but
requires a paid subscription and holds the same file; use GitHub.

**Cite:** S. C. Hassler, U. A. Mughal and M. Ismail, "Cyber-physical intrusion
detection system for unmanned aerial vehicles," *IEEE Trans. Intell. Transp.
Syst.*, vol. 25, no. 6, pp. 6106–6117, Jun. 2024, doi: 10.1109/TITS.2023.3339728.
The BibTeX in the repository README gives a different, incorrect DOI.

### It is not one table

It is five exports concatenated and padded to 38 comma-separated fields. Reading
it with a single header — what a plain `pd.read_csv` does — silently produces
wrong data: the label for every physical row lands in the column named
`ip.flags`, and the labels for evil twin and false data injection disappear
entirely. 21,672 of 54,774 data rows come back with a null class.

| Segment | Class | Cyber rows | Cyber features | Physical rows | Physical features |
|---|---|---|---|---|---|
| 0 | benign | 9,425 | 37 | 4,290 | 16 |
| 1 | DoS | 11,671 | 37 | 973 | 16 |
| 2 | replay | 12,006 | 37 | 973 | 16 |
| 3 | evil twin | 5,683 | **34** | 5,473 | **21** |
| 4 | FDI | 3,473 | **34** | 807 | **31** |

Segments 0–2 carry the 37 + 16 schema the releasing paper describes. Segments 3
and 4 were captured with different instrumentation: their cyber header has a
different column order and adds `wlan_radio.signal_strength`, `Noise level`,
`SNR`, `preamble`; their physical view is the DJI Tello SDK field set, and FDI
adds `mpitch, mroll, myaw, est_x, est_y, cntl_x, cntl_y, residual1..4`. Each
segment opens with its own `timestamp_c,…` header and carries a second header row
starting `timestamp_p` — or `mid` in the last segment — separating its cyber
block from its physical block.

`release/` in this repository holds a corrected repackaging: ten files, one per
block, each with the header that block actually has, verified cell by cell
against the source. Use `release/uav_cpids.py` rather than writing a reader.

## 2. MAVLink message identifiers (GUIDE)

Sequences of MAVLink message identifiers from hardware-in-the-loop missions under
heartbeat, ping and request flooding. The class is carried in the file name, not
in the row.

```bash
git clone https://github.com/hcrlab-knu/GUIDE     # path may change; see the paper
# pass the directory holding the raw sequence files
```

**Cite:** J. D. Yoo, H. Kim and H. K. Kim, "GUIDE: GAN-based UAV IDS
enhancement," *Comput. Secur.*, vol. 147, 104073, 2024,
doi: 10.1016/j.cose.2024.104073.

**What the adapter reads.** One capture per file. The label comes from the file
name, so no per-record attack type exists and check B4 fails. Two captures are
labelled normal, which is what makes the leave-one-capture-out result on this
corpus unstable: removing one of them removes half the class.

## 3–4. UAVCAN attack datasets, 2022 and 2026 (LUMI)

DroneCAN bus traffic across ten scenarios combining flooding, fuzzing and replay.
Both releases interleave normal and attack frames inside every capture, which is
why they score lowest on the fingerprint index and fail the benign-only-capture
check.

Obtain from the Hacking and Countermeasure Research Lab, Korea University:
<https://ocslab.hksecurity.net/Datasets/uavcan-attack-dataset> and
<https://ocslab.hksecurity.net/Datasets/uavcan-attack-dataset-2026-lumi>.
Registration is free; the files cannot be fetched without a browser.

**Cite:** D. Kim, Y. Song, Y. Kwon, H. Kim, J. D. Yoo and H. K. Kim, "UAVCAN
dataset description," arXiv:2212.09268, 2022; and Y. Song and H. K. Kim, "UAVCAN
attack dataset 2026 (LUMI)," HCRL, Korea University, 2026.

**What the adapter reads.** The 2022 release ships `.bin` files with no header
row, no commas, and SocketCAN `candump` text with a label word prepended; the
data field holds one hexadecimal byte per unit of the length code, so the
whitespace-separated field count varies between six and thirteen across rows. A
comma-separated read does not fail — it returns a single column of strings — so
the format defect is invisible unless the file is opened. The 2026 release is
comma-separated with one consistent header across all ten files.

## 5. SHADOW-GCS

MAVLink sessions under a spoofed ground control station, crossed over transport
path, vehicle state and class: 35 captures, 641,015 packets, three to ten
captures in every cell of the design, fourteen of them benign throughout. This is
the UAV corpus that satisfies the design requirement the paper argues for.

<https://ocslab.hksecurity.net/Datasets/shadow-gcs>, or IEEE DataPort
doi `10.21227/czt3-y366`.

**Cite:** J. Y. Lee and H. K. Kim, "SHADOW-GCS attack dataset," IEEE DataPort,
2026, doi: 10.21227/czt3-y366.

**What the adapter reads.** Eleven shallow features over windows of 64 packets —
length and inter-arrival statistics, packet rate, byte rate — with no payload, no
addresses and no ports. The shallowness is deliberate: it makes the fingerprint
measurement a lower bound rather than an artefact of a rich representation.

## 6. HAI 21.03 — HIL-based augmented ICS security dataset

Included so the audit is not confined to one application domain, and so the
design requirement can be tested against a corpus built in a different field.
Eight captures from one testbed at one row per second: three with no attack
running and five carrying attacks, 1,323,608 rows in total.

**Download.** The official repository serves the newer releases through Git LFS,
and that budget is presently exhausted, so `git lfs pull` fails with
`This repository exceeded its LFS budget`. HAI 21.03 predates that and is stored
as ordinary git objects, so a plain clone retrieves it:

```bash
git clone https://github.com/icsdataset/hai            # 21.03 arrives as .csv.gz
cd hai/hai-21.03 && gunzip -k *.csv.gz
```

If the clone stops on an LFS error for a newer release, the 21.03 files are
already on disk and usable; the failure concerns `hai-22.04` and `hai-23.05`
only. A Kaggle copy of the same release is published by the same group at
<https://www.kaggle.com/datasets/icsdataset/hai-security-dataset>.

**Cite:** H.-K. Shin, W. Lee, J.-H. Yun and B.-G. Min, "Two ICS security datasets
and anomaly detection contest on the HIL-based augmented ICS testbed," in *Proc.
Cyber Security Experimentation and Test Workshop (CSET)*, 2021, pp. 36–40,
doi: 10.1145/3474718.3474719.

**What the adapter reads.** 79 process points per row; `time` and the four
`attack*` columns are excluded from the features. The overall `attack` column is
the label and the file is the capture, which is the granularity the release
itself distinguishes. Attack rows are 0.7 % of the corpus, so accuracy is
saturated by the majority class and `measure_split_cost.py` reports balanced
accuracy alongside it. HAI is also the only corpus in the set whose release
states how to split for evaluation, so it is the one column where check D3
passes.

---

## After downloading

```bash
python3 audit/run_checklist.py \
    --hassler data/hassler/Dataset_T-ITS.csv \
    --mavlink /path/GUIDE/data/raw \
    --uavcan  /path/uavcan_extracted \
    --lumi    "/path/UAVCAN Attack Dataset 2026 (LUMI)" \
    --shadow  /path/SHADOW-GCS \
    --hai     /path/hai/hai-21.03 \
    --out runs/audit/checklist.json
```

Six corpora in one process peaks above 7 GB of memory. On a machine with less,
run one corpus per process and merge the resulting objects; each writes its own
top-level key, so merging is a dictionary update.
