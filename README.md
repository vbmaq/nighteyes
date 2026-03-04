# Glint Constellation Pipeline (`paperReady`)

This repository contains the Glint Constellation pipeline: a set of UIs plus the underlying per-image detection and constellation-matching code. 

For a quick start, see **Install** and **Run** below.

## Repository Layout
- `glint_pipeline/`: core pipeline + small helper modules (importable)
- `apps/`: UI implementations
- `tools/`: plotting / dataset conversion utilities
- `docs/`: pipeline documentation
- `templates/`: default template bank (JSON)
- `data/`: results / templates / example assets

## Entry Points (Backwards Compatible)
The following scripts are thin wrappers (to preserve older entry points):
- Preview/tuning UI: `glint_pipeline_preview_ui.py` (wrapper -> `apps/`)
- Annotation review/correction UI: `glint_pupil_annotation_review_ui.py` (wrapper -> `apps/`)
- Manual template maker UI: `templatemaker_manual.py` (wrapper -> `apps/`)
- Core pipeline (CLI + library): `glint_pipeline_eval_gen.py` (wrapper -> `glint_pipeline/`)

## Data and Templates
- Dataset / results are kept under `data/`.
- Default template bank is loaded from `templates/default_templates.json` when `--template_bank_source default` is used.
  - If you use a custom bank, pass `--template_bank_source custom --template_bank_path <path>`.

## Install
Use an existing environment, or install the minimal dependencies:
```bash
python -m pip install -r requirements.txt
```

## Run
Preview UI:
```bash
python glint_pipeline_preview_ui.py
```

Annotation review/correction UI:
```bash
python glint_pupil_annotation_review_ui.py --help
```

Manual template maker:
```bash
python templatemaker_manual.py
```

Eval CLI (example):
```bash
python glint_pipeline_eval_gen.py <image_folder_or_dataset_root> --help
```
