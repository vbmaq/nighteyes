@echo off
REM Tier B3 sweep (SLA ablation). Outputs: annotated\sweeps\tierB_b3_summary.csv
python glint_pipeline_eval.py data\LabelledImages\dataset --labels data\label.txt --sweep --sweep_grid sweeps\tierB_b3.json --sweep_out_csv annotated\sweeps\tierB_b3_summary.csv --sweep_id tierB\b3 --sweep_keep_reports
