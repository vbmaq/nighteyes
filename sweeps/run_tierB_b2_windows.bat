@echo off
REM Tier B2 sweep (matcher-specific tuning). Outputs: annotated\sweeps\tierB_b2_summary.csv
python glint_pipeline_eval.py data\LabelledImages\dataset --labels data\label.txt --sweep --sweep_grid sweeps\tierB_b2.json --sweep_out_csv annotated\sweeps\tierB_b2_summary.csv --sweep_id tierB\b2 --sweep_keep_reports --sweep_skip_existing
