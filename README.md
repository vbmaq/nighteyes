# Glint Constellation Pipeline

Reference implementation of a constellation-based corneal reflection ("glint") detection and correspondence pipeline for multi-LED eye-tracking systems.

This repository accompanies the NightEyes framework and provides a reproducible implementation of the candidate detection, scoring, and geometric matching pipeline used for identity-preserving glint correspondence.

## What's Included

- Modular detection and matching pipeline
- Interactive UIs for template creation and annotation review
- Evaluation scripts for benchmarking correspondence accuracy
- Utilities for dataset processing and visualization

## Overview

Video-based eye trackers using the pupil-corneal reflection (P-CR) method estimate gaze by detecting the pupil and reflections of infrared LEDs on the cornea. Accurate gaze estimation requires reliable localization and identity-preserving correspondence between detected glints and the physical LEDs.

Glint correspondence is challenging due to:

- Spurious reflections
- Missing glints due to occlusion
- Intensity variation and shape distortion
- Varying illumination conditions

This repository implements a constellation-based matching pipeline inspired by star identification algorithms, where LED constellations are matched to detected glint constellations under similarity transforms.

### Pipeline Stages

1. Candidate detection
2. Candidate scoring
3. Geometric constellation matching
4. Identity assignment and evaluation

## Repository Layout

- `glint_pipeline/`: Core pipeline implementation and matching algorithms
- `apps/`: Interactive UIs for inspection, annotation, and template creation
- `tools/`: Utilities for dataset conversion, plotting, and evaluation
- `docs/`: Additional documentation
- `templates/`: Default LED constellation templates
- `data/`: Results, templates, and example assets (recommended)

## Entry Points

The following scripts provide the primary entry points for the pipeline:

- `glint_pipeline_eval_gen.py`: Run the full detection and matching pipeline on a dataset
- `glint_pipeline_preview_ui.py`: Preview and parameter tuning UI
- `glint_pupil_annotation_review_ui.py`: Annotation review and correction UI
- `templatemaker_manual.py`: Manual template creation UI

## Installation

Install dependencies using pip:

```bash
python -m pip install -r requirements.txt
```

Notes:

- Python 3.9+ is recommended.
- UI tools use Tkinter (included with many Python distributions; on some platforms it may require an additional system package).

## Running

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

Evaluation CLI (example):

```bash
python glint_pipeline_eval_gen.py <dataset_folder> --help
```

## Templates

The constellation matching stage requires a template describing the relative LED arrangement.

Default templates are stored in `templates/default_templates.json`.

To use a custom template bank, pass:

```bash
python glint_pipeline_eval_gen.py <dataset_folder> --template_bank_source custom --template_bank_path <path>
```

## Outputs

The pipeline commonly produces outputs such as:

```text
annotated/  # Visual overlays showing candidates and matched glints
reports/    # Matching accuracy and localization statistics
csv/        # Evaluation metrics for downstream analysis
```

## Reproducibility

This repository includes the components used for reproducible evaluation and benchmarking of glint matching algorithms, including pipeline stages, template tools, and evaluation scripts. Hyperparameter sweeps and configuration selection are tracked in the repository (see `sweeps/`).

## Citation

If you use this code in academic work, please cite the associated paper:

> Paper citation placeholder

## License

License is currently unspecified. Add a `LICENSE` file to clarify permitted use.

## Acknowledgements

The project is supported by the Chips Joint Undertaking (Chips JU) and its members, including top-up funding by
Denmark, Germany, Netherlands, Sweden, under grant agreement No. 101139942.

This repository builds on ideas from constellation matching and star identification algorithms used in spacecraft attitude estimation.

