#!/usr/bin/env bash
set -euo pipefail
# Tier B sweep (end-to-end). Outputs: annotated/sweeps/tierB/<run_id> and annotated/sweeps/tierB_summary.csv
python glint_pipeline_eval.py data/LabelledImages/dataset --labels data/label.txt --sweep --sweep_grid sweeps/tierB_end_to_end.json --sweep_out_csv tierB_summary.csv --sweep_id tierB/run --sweep_keep_reports
