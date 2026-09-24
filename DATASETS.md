# Where the data comes from, and what is actually in it

Every path below was checked against the live source. Two of the three public
corpora need a free account; none of them can be fetched without a browser
except the first.

---

## 1. Hassler et al. — cyber-physical corpus (primary)

The only public UAV corpus with both a network view and a flight view from one
testbed.

**Download (no account needed):**

```bash
mkdir -p data/hassler
curl -L -o data/hassler/Dataset_T-ITS.csv \
  https://raw.githubusercontent.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks/main/Dataset_T-ITS.csv
# 6,148,114 bytes, 54,784 lines including headers
```

Repository: <https://github.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks>
(MIT licence). There is also an IEEE DataPort mirror, doi `10.21227/6f22-py65`,
but it **requires a paid DataPort subscription** and holds the same file. Use
GitHub.

**Cite:** S. C. Hassler, U. A. Mughal and M. Ismail, "Cyber-physical intrusion
detection system for unmanned aerial vehicles," *IEEE Trans. Intell. Transp.
Syst.*, vol. 25, no. 6, pp. 6106–6117, Jun. 2024, doi: 10.1109/TITS.2023.3339728.
The authors' own BibTeX in the repository README gives a different, incorrect
DOI — use the one above.

### What the file really contains

**It is not one table.** It is five separate exports concatenated and padded to
38 comma-separated fields. Reading it with a single header — which is what a
plain `pd.read_csv` does — silently produces wrong data: the label for every
physical row lands in the column named `ip.flags`, and the labels for evil twin
and FDI disappear entirely (21,679 of 54,783 rows come back with a null class).

Verified structure:

| Segment | Class | Cyber rows | Cyber features | Physical rows | Physical features |
|---|---|---|---|---|---|
| 0 | benign | 9,425 | 37 | 4,290 | 16 |
| 1 | DoS | 11,671 | 37 | 973 | 16 |
| 2 | replay | 12,006 | 37 | 973 | 16 |
| 3 | evil twin | 5,683 | **34** | 5,473 | **21** |
| 4 | FDI | 3,473 | **34** | 807 | **31** |

Segments 0–2 carry the 37 + 16 schema the paper describes. Segments 3 and 4 were
captured with different instrumentation: their cyber header has a different
column order and adds `wlan_radio.signal_strength`, `Noise level`, `SNR`,
`preamble`; their physical view is the DJI Tello SDK field set
(`mid, x, y, z, pitch, roll, yaw, vgx, vgy, vgz, templ, temph, tof, h, bat,
baro, time, agx, agy, agz`), and FDI adds `mpitch, mroll, myaw, est_x, est_y,
cntl_x, cntl_y, residual1..4`.

Each segment begins with its own `timestamp_c,...` header and contains a second
header row starting `timestamp_p` (or `mid` in the last segment) that separates
its cyber block from its physical block.

**Check your copy before using it:**

```bash
python -c "from clids import hassler; print(hassler.describe('data/hassler/Dataset_T-ITS.csv'))"
```

`clids/hassler.py` handles all of this. Two policies:

- `schema: strict` — the three classes sharing the documented 37 + 16 schema.
  35 cyber and 15 physical features after dropping timestamps and frame
  numbers; 33,102 rows; **two attack families**, so only two leave-one-attack-out folds.
- `schema: common` — all five classes on the intersection of normalized feature
  names. 12 cyber and 12 physical features; 42,258 rows; **four attack
  families**. The physical intersection is
  `barometer, battery, distance, flight_time, height, pitch, roll, temperature,
  x_speed, y_speed, yaw, z_speed`.

Whichever you pick, say so in the paper and give the feature counts. A reviewer
who has opened this file will know the difference.

### The size problem — read this before planning experiments

The corpus has 9,425 benign rows. At the paper's window length that is not
enough benign data to fit a tail:

| T | stride | total windows | benign windows | benign for calibration at 25 % |
|---|---|---|---|---|
| 16 | 8 | 5,258 | 1,173 | 293 |
| 32 | 16 | 2,618 | 584 | 146 |
| 64 | 32 | 1,296 | 289 | 72 |
| 128 | 64 | 636 | 142 | 35 |
| 128 | 12 (90 % overlap) | 3,043 | 678 | 169 |

A 1 % threshold needs on the order of 2,000 benign windows for the estimate to
be stable, and 0.1 % needs roughly ten times that — `TailFit.required_calibration_windows`
computes the exact figure from equation (14) for your data. **The Hassler corpus
is short by about an order of magnitude.** Pushing the overlap to 90 % inflates
the window count without adding information, and correlated excesses break the
independence the tail fit assumes, so it does not fix the problem.

Consequences for the plan: this corpus supports the fusion comparison, but the
benign mass for the calibration study has to come from somewhere else. The
synchronized corpus below is the way out, and it is now load-bearing rather than
a supplement.

---

## 2. Whelan et al. — UAV Attack Dataset (real-hardware telemetry)

PX4 flight logs from a Pixhawk 4 on a Holybro S500, under live GNSS spoofing
generated with a HackRF, live jamming, and MAVLink ping-flood DoS.

- Page: <https://ieee-dataport.org/open-access/uav-attack-dataset>
- DOI: `10.21227/00dg-0d12`
- **Open access**, but a free IEEE account is required to download. There is no
  direct link; sign in and click `UAVAttackData.zip` (683.88 MB).

```bash
mkdir -p data/uav_attack
# unzip the downloaded archive here; .ulg and .csv are both handled
unzip ~/Downloads/UAVAttackData.zip -d data/uav_attack/
pip install pyulog     # only if the archive contains .ulg rather than .csv
```

**Cite:** J. Whelan, T. Sangarapillai, O. Minawi, A. Almehmadi and K. El-Khatib,
"Novelty-based intrusion detection of sensor attacks on unmanned aerial
vehicles," in *Proc. 16th ACM Symp. QoS Security Wireless Mobile Netw.
(Q2SWinet)*, Alicante, Spain, Nov. 2020, pp. 23–28, doi: 10.1145/3416013.3426446.

This corpus has **no packet capture**. It exercises the physical branch and the
degraded-modality path, not fusion. `clids/data.py:load_px4` derives the family
from the file name; check the mapping in `data.px4.family_from_filename` against
the archive's actual layout after unzipping.

---

## 3. ISOT drone corpus (cross-testbed generalization)

Network captures from a DJI Tello testbed — a different airframe in a different
environment, which is what makes it useful as an unseen-environment test.

- Page: <https://onlineacademiccommunity.uvic.ca/isot/2024/12/05/drone-datasets/>
- A direct Google Drive link is on that page ("Click here to download the ISOT
  Drone Anomaly Detection Dataset"). Over 23 GB of PCAP plus 1.3 GB of extracted
  CSV features; take the CSV unless you intend to re-extract.
- Scripts: <https://github.com/isot-lab/Drone-Anomaly-Detection-Dataset-and-Unsupervised-Machine-Learning>
- No licence is stated. Contact `traore at ece.uvic.ca` before redistributing
  anything derived from it.

```bash
mkdir -p data/isot
# place the extracted feature CSV here, then point the config at it
```

**Cite:** Z. Chen, I. Traoré, M. Mamun and S. Saad, "Drone anomaly detection:
Dataset and unsupervised machine learning," in *Foundations and Practice of
Security* (LNCS 15532). Cham: Springer, 2025, pp. 186–201,
doi: 10.1007/978-3-031-87499-4_12.

---

## 4. Synchronized cyber-physical corpus (generated here)

No public corpus records network traffic and flight telemetry from the same
sessions with one clock. Given the size problem in §1, generating one is not
optional for this paper — it is where the benign calibration mass comes from.

The setup couples a flight simulator to a network simulator over a shared time
base:

- ArduPilot SITL — <https://ardupilot.org/dev/docs/sitl-simulator-software-in-the-loop.html>
- ns-3 — <https://www.nsnam.org/releases/>
- A published harness that already bridges the two over ZeroMQ and emits
  synchronized PCAP and telemetry CSV, with eight scripted attack scenarios:
  <https://github.com/Wh02m1/UAVLnQ>

Plan on several hours of benign flight. At T = 128 with 50 % overlap and a
50 Hz telemetry rate, one hour of benign flight yields roughly 2,800 windows, so
four to six hours puts the calibration set comfortably past what equation (14)
requires at α = 10⁻³.

Release the generated corpus under CC BY on Zenodo or IEEE DataPort with the
paper. Given what §1 documents about the state of the only existing paired
release, a clean synchronized corpus is likely to be the most cited thing in
this work.

---

## 5. Optional, for pretraining only

- **CICIoT2023** — <https://www.unb.ca/cic/datasets/iotdataset-2023.html>, public.
  Cite Neto et al., *Sensors* 23(13):5941, 2023, doi: 10.3390/s23135941.
- **Edge-IIoTset** — IEEE DataPort doi `10.21227/mbc1-1h68` (subscription); the
  author also mirrors it on Kaggle. Cite Ferrag et al., *IEEE Access* 10:40281–40306, 2022.
- **ALFA** — <http://theairlab.org/alfa-dataset>. Flight telemetry with
  **faults, not attacks**. Usable to pretrain the telemetry encoder; calling it
  an attack dataset would be wrong.

---

## Two names that appear in the literature and should not be used

- **"Drone-CyberAttack" / "Drone-CyberAttack-2024" does not exist.** No
  repository, no release, no source paper. If a draft cites it, the citation is
  fabricated.
- **"UAV-IDS-2020" is not an attack dataset.** It is the UAV *presence
  detection* corpus of Alipour-Fanid et al. over encrypted Wi-Fi (UCI ID 564,
  doi `10.24432/C56P6X`), and it contains no attack classes at all. Several
  published papers treat it as an intrusion-detection benchmark; they are wrong,
  and repeating the error is an easy thing for a reviewer to catch.

---

## Checklist before the first training run

```bash
# 1. structure of the primary corpus, checked against the README
python -c "from clids import hassler; print(hassler.describe('data/hassler/Dataset_T-ITS.csv'))"

# 2. windows, class counts and the corpus fingerprint
python -m clids.cli prepare -c configs/hassler.yaml

# 3. how much benign data the tail fit actually needs for your data
python - <<'EOF'
from clids import evt
import numpy as np
# after a first training run, substitute real benign scores here
scores = np.random.gamma(2.0, 1.0, 5000)
f = evt.fit_tail(scores, percentile=95.0)
for a in (1e-2, 1e-3):
    print(f"alpha={a:g}: need {f.required_calibration_windows(a)} benign windows")
EOF
```

If step 3 reports more windows than step 2 produced, the calibration study
cannot be run on that corpus alone. That is the situation with the Hassler
corpus today, and §4 is the answer to it.

---

## 6. HAI 21.03 — HIL-based augmented ICS security dataset

Included so that the audit is not confined to one application domain, and so
that the design requirement the paper argues for can be tested against a corpus
built in a different field. Eight captures from one testbed at one row per
second: three recorded with no attack running, five carrying attacks.

**Download.** The official repository serves the newer releases through Git LFS,
and that budget is presently exhausted, so `git lfs pull` fails with
`This repository exceeded its LFS budget`. HAI 21.03 predates that and is stored
as ordinary git objects, so a plain clone retrieves it:

```bash
git clone https://github.com/icsdataset/hai            # 21.03 arrives as .csv.gz
cd hai/hai-21.03 && gunzip -k *.csv.gz
```

If the clone stops on an LFS error for a newer release, the 21.03 files are
already on disk and usable; the checkout failure concerns `hai-22.04` and
`hai-23.05` only. A Kaggle copy of the same release is published by the same
group at <https://www.kaggle.com/datasets/icsdataset/hai-security-dataset>.

**What the adapter reads.** 79 process points per row; `time` and the four
`attack*` columns are excluded from the features. The overall `attack` column
is the label, and the file is the capture, which is the granularity the release
itself distinguishes. Attack rows are 0.7 % of the corpus, so accuracy is
saturated by the majority class and `measure_split_cost.py` reports balanced
accuracy alongside it.

**Cite:** H.-K. Shin, W. Lee, J.-H. Yun and B.-G. Min, "Two ICS security
datasets and anomaly detection contest on the HIL-based augmented ICS testbed,"
in *Proc. Cyber Security Experimentation and Test Workshop (CSET)*, 2021,
pp. 36–40, doi: 10.1145/3474718.3474719.
