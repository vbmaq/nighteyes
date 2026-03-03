# Glint Constellation Pipeline (paperReady)

This branch is a publication-oriented artifact focused on the glint pipeline UIs and the underlying per-image detection + constellation matching pipeline.

## Intent-Based Layout
- `glint_pipeline/`: core pipeline + small helper modules (importable)
- `apps/`: UI implementations
- `tools/`: plotting / dataset conversion utilities
- `docs/`: pipeline documentation
- `templates/`: default template bank (JSON)
- `data/`: results / templates / example assets

## Entry Points (Backwards Compatible)
- Preview/tuning UI: `glint_pipeline_preview_ui.py` (wrapper -> `apps/`)
- Annotation review/correction UI: `glint_pupil_annotation_review_ui.py` (wrapper -> `apps/`)
- Manual template maker UI: `templatemaker_manual.py` (wrapper -> `apps/`)
- Core pipeline (CLI + library): `glint_pipeline_eval_gen.py` (wrapper -> `glint_pipeline/`)

## Data And Templates
- Dataset / results are kept under `data/`.
- Default template bank is loaded from `templates/default_templates.json` when `--template_bank_source default` is used.
  - If you use a custom bank, pass `--template_bank_source custom --template_bank_path <path>`.

## Install
Use your existing environment, or install minimal dependencies:
```bash
python -m pip install -r requirements-paperready.txt
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
