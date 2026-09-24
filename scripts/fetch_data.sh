#!/usr/bin/env bash
# Fetch what can be fetched without a browser, and say plainly what cannot.
# See DATASETS.md for the full picture.
set -euo pipefail
cd "$(dirname "$0")/.."

HASSLER_URL="https://raw.githubusercontent.com/uamughal/UAVs-Dataset-Under-Normal-and-Cyberattacks/main/Dataset_T-ITS.csv"
HASSLER_OUT="data/hassler/Dataset_T-ITS.csv"

echo "== 1/3  Hassler et al. cyber-physical corpus (no account needed)"
mkdir -p data/hassler
if [ -s "$HASSLER_OUT" ]; then
  echo "   already present: $HASSLER_OUT ($(wc -c <"$HASSLER_OUT") bytes)"
else
  curl -fL --retry 3 -o "$HASSLER_OUT" "$HASSLER_URL"
  echo "   downloaded $(wc -c <"$HASSLER_OUT") bytes, $(wc -l <"$HASSLER_OUT") lines"
fi
echo "   checking structure ..."
python -c "from clids import hassler; print(hassler.describe('$HASSLER_OUT'))" || {
  echo "   !! the file does not parse as expected. Compare the output above"
  echo "      with the authors' README before going further."
  exit 1
}

echo
echo "== 2/3  UAV Attack Dataset (Whelan et al.) - manual step"
mkdir -p data/uav_attack
if compgen -G "data/uav_attack/*" > /dev/null; then
  echo "   something is already in data/uav_attack/"
else
  cat <<'MSG'
   This one needs a free IEEE account; there is no direct link.
     1. open https://ieee-dataport.org/open-access/uav-attack-dataset
     2. sign in (or create a free IEEE account)
     3. download UAVAttackData.zip (683.88 MB)
     4. unzip ~/Downloads/UAVAttackData.zip -d data/uav_attack/
   If the archive holds .ulg rather than .csv:  pip install pyulog
MSG
fi

echo
echo "== 3/3  ISOT drone corpus - manual step"
mkdir -p data/isot
if compgen -G "data/isot/*" > /dev/null; then
  echo "   something is already in data/isot/"
else
  cat <<'MSG'
   Direct Google Drive link, from:
     https://onlineacademiccommunity.uvic.ca/isot/2024/12/05/drone-datasets/
   Take the 1.3 GB feature CSV rather than the 23 GB of PCAP unless you intend
   to re-extract features. Put it at data/isot/drone_flows.csv
   No licence is stated; contact traore at ece.uvic.ca before redistributing.
MSG
fi

echo
echo "Next:  python -m clids.cli prepare -c configs/hassler.yaml"
echo "Read DATASETS.md first - the primary corpus has a size limit that affects"
echo "the experimental plan."
