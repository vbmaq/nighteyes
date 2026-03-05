# Night Eyes Glint Constellation Pipeline

Reference implementation of a constellation-based corneal reflection ("glint") detection and correspondence pipeline for multi-LED eye-tracking systems.

This repository accompanies the NightEyes framework and provides a reproducible implementation of the candidate detection, scoring, and geometric matching pipeline used for identity-preserving glint correspondence.

Pre-generated annotations for the OpenEDS datasets [1,2] can be found in https://zenodo.org/records/18860585. 

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

1. Image enhancement
2. Candidate detection
3. Candidate scoring
4. Geometric constellation matching
5. Identity assignment and evaluation

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

## Running on Chugh et al. dataset 

To run the pipeline on the Chugh et al.[3] dataset, first download their dataset here: https://www.eecg.utoronto.ca/~jayar/datasets/xreyetrack.html

Then run the command below (the default template comes pre-loaded, just load the image folder containing Chugh et al.'s dataset. You can also find sample frames from the dataset in `data/chugh_samples`.   
```bash
python glint_pipeline_preview_ui.py
```

## Running general
Review and correct annotations by using:
```bash
python glint_pupil_annotation_review_ui.py --help
```
Pre-generated annotations can be found in the Zenodo repository (see above)


Manual template maker -- if trying out on a different dataset, you can create your template by loading an example image from the dataset that shows all glints you need to detect and manually assigning each glint with a label: 

```bash
python templatemaker_manual.py
```

Preview UI with winning sweep config pre-loaded -- currently slow. As a shortcut, save the glintz directly and use the annotation review ui. 

```bash
python glint_pipeline_preview_ui.py
```

Annotation review/correction UI:

```bash
python glint_pupil_annotation_review_ui.py --help
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


## To-Do's 
- optimize glint UI
- add stereo capabilities
- improve Correction mode
- improve built-in pupil detection methods

## Reproducibility

This repository includes the components used for reproducible evaluation and benchmarking of glint matching algorithms, including pipeline stages, template tools, and evaluation scripts. Hyperparameter sweeps and configuration selection are tracked in the repository (see `sweeps/`).

## Citation

If you use this code in academic work, please cite the associated paper:

> Paper citation placeholder

## License

License is currently unspecified.

## Acknowledgements

The project is supported by the Chips Joint Undertaking (Chips JU) and its members, including top-up funding by
Denmark, Germany, Netherlands, Sweden, under grant agreement No. 101139942.

This repository builds on ideas from constellation matching and star identification algorithms used in spacecraft attitude estimation.


## Literature
* [1] Garbin, S. J., Shen, Y., Schuetz, I., Cavin, R., Hughes, G., & Talathi, S. S. (2019). Openeds: Open eye dataset. arXiv preprint arXiv:1905.03702.
* [2] Palmero, C., Sharma, A., Behrendt, K., Krishnakumar, K., Komogortsev, O. V., & Talathi, S. S. (2020). Openeds2020: Open eyes dataset. arXiv preprint arXiv:2005.03876.
* [3] Chugh, S., Brousseau, B., Rose, J., & Eizenman, M. (2021, January). Detection and correspondence matching of corneal reflections for eye tracking using deep learning. In 2020 25th international conference on pattern recognition (ICPR) (pp. 2210-2217). IEEE.

  
