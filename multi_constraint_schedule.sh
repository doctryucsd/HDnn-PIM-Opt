#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEDULE_SCRIPT="${SCRIPT_DIR}/constraint_schedule.sh"
RANDOM_SCRIPT="${SCRIPT_DIR}/random.sh"

: "${DATASET:=ucihar}"

if [ ! -x "${SCHEDULE_SCRIPT}" ]; then
  if [ -f "${SCHEDULE_SCRIPT}" ]; then
    chmod +x "${SCHEDULE_SCRIPT}"
  else
    echo "Could not find constraint_schedule.sh next to this script" >&2
    exit 1
  fi
fi

if [ ! -x "${RANDOM_SCRIPT}" ]; then
  if [ -f "${RANDOM_SCRIPT}" ]; then
    chmod +x "${RANDOM_SCRIPT}"
  else
    echo "Could not find random.sh next to this script" >&2
    exit 1
  fi
fi

acqfs=("qExpectedHypervolumeImprovement" "qNoisyExpectedHypervolumeImprovement")

for acqf in "${acqfs[@]}"; do
  echo "Launching runs for ACQF=${acqf} with CONSTRAINT=false"
  DATASET="${DATASET}" CONSTRAINT=false POLICY=static ACQF="${acqf}" bash "${SCHEDULE_SCRIPT}"

  echo "Launching runs for ACQF=${acqf} with CONSTRAINT=true, POLICY=static"
  DATASET="${DATASET}" CONSTRAINT=true POLICY=static ACQF="${acqf}" bash "${SCHEDULE_SCRIPT}"

  echo "Launching runs for ACQF=${acqf} with CONSTRAINT=true, POLICY=linear"
  DATASET="${DATASET}" CONSTRAINT=true POLICY=linear ACQF="${acqf}" bash "${SCHEDULE_SCRIPT}"
done

# echo "Launching random baseline runs"
# DATASET="${DATASET}" bash "${RANDOM_SCRIPT}"
