#!/usr/bin/env bash
set -euo pipefail
# Tier A sweep (matcher-only). Outputs: annotated/sweeps/tierA/<run_id> and annotated/sweeps/tierA_summary.csv
python glint_pipeline_eval.py data/LabelledImages/dataset --labels data/label.txt --sweep --sweep_grid sweeps/tierA_matcher_only.json --sweep_out_csv tierA_summary.csv --sweep_id tierA/run --sweep_keep_reports
