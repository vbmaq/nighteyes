@echo off
REM Tier B1 sweep (fair head-to-head). Outputs: annotated\sweeps\tierB_b1_summary.csv
python glint_pipeline_eval.py data\LabelledImages\dataset --labels data\label.txt --sweep --sweep_grid sweeps\tierB_b1.json --sweep_out_csv annotated\sweeps\tierB_b1_summary.csv --sweep_id tierB\b1 --sweep_keep_reports
