#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

RESULT_DIR="${ROOT_DIR}/results"
METRICS_DIR="${RESULT_DIR}/metrics"
TABLES_DIR="${RESULT_DIR}/tables"
FIGURES_DIR="${RESULT_DIR}/figures"

mkdir -p "${METRICS_DIR}" "${TABLES_DIR}" "${FIGURES_DIR}" "${FIGURES_DIR}/extra"

DATASETS=(mnist fashion cifar10)
FLAGSHIP_METHOD="NEHVI_no-constraint"
EARLY_COUNT=30
FILTER_KS=(10 20 30)

for dataset in "${DATASETS[@]}"; do
  echo "Processing ${dataset}"
  python3 -m analysis.rolling_eval --dataset "${dataset}" --out "${METRICS_DIR}/${dataset}_proxy_metrics.csv"
  python3 -m analysis.filter_efficacy --dataset "${dataset}" --out "${TABLES_DIR}/${dataset}_filter.csv" --k "${FILTER_KS[@]}"
  python3 -m analysis.reorder_counterfactual --dataset "${dataset}" --early "${EARLY_COUNT}" --out "${TABLES_DIR}/${dataset}_counterfactual.csv"
  python3 -m analysis.plotting pred_true --dataset "${dataset}" --out "${FIGURES_DIR}/${dataset}_pred_true.pdf"
  python3 -m analysis.plotting early_eff --dataset "${dataset}" --method "${FLAGSHIP_METHOD}" --early "${EARLY_COUNT}" --out "${FIGURES_DIR}/${dataset}_early_eff.pdf"
  python3 -m analysis.plotting filter_bars --dataset "${dataset}" --table "${TABLES_DIR}/${dataset}_filter.csv" --out "${FIGURES_DIR}/${dataset}_filter_bars.pdf"
  echo "Finished ${dataset}"
  echo
done

echo "All analyses completed." 
