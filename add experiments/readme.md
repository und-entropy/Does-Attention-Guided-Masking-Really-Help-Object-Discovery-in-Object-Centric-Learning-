# Additional experiments

This directory contains only the six-dataset RM/AGM primary comparison and the figures and tables used by the workshop paper. The files below are listed by function.

## Python files

| File | Purpose |
|---|---|
| `tables_AGM.py` | Recomputes the main and absolute-result tables from the 36 primary evaluation records, including means, sample standard deviations, and paired AGM-RM differences. `--check_paper` checks the final paper tables and the clean evaluation-log excerpt. |
| `benchmark_AGM.py` | Re-evaluates the six-dataset, two-strategy, three-seed primary comparison with supplied datasets, checkpoints, and training logs. It does not run additional ablations. |
| `common_AGM.py` | Shared utilities for configuration parsing, log inspection, checkpoint selection, dataset/model construction, evaluation, and first-frame result normalization. It is imported by the evaluation scripts and is not a standalone entry point. |
| `capture_actual_masks_AGM.py` | Captures RM/AGM segmentations and the actual decoder patch masks for the paper's selected samples. Both methods use a fixed 50% masking budget. This script requires datasets and checkpoints. |
| `plot_actual_masks_AGM.py` | Builds the appendix-style six-column segmentation and actual-mask comparison figures from the captured arrays. |
| `plot_main_qualitative_AGM.py` | Reassembles the two-sample qualitative figure used in the main paper from the final PDFs in `../res/`. It does not require model weights. |
| `plot_seed_effects_AGM.py` | Plots per-seed AGM-RM differences for ARI, FG-ARI, mBO, and mIoU from `data/primary_AGM.csv`. |
| `plot_training_curves_AGM.py` | Plots the historical RM/AGM ARI and FG-ARI validation curves from `data/training_curves_AGM.csv`; it does not synthesize or interpolate metrics. |

## Data and paper materials

| Path | Purpose |
|---|---|
| `data/primary_AGM.csv` | The 36-model primary comparison used by the paper tables and seed-effect plot. |
| `data/training_curves_AGM.csv` | Historical validation records used for the training-curve figure. |
| `data/clean_evaluation_AGM.txt` | A compact excerpt of the clean primary evaluation records used for audit checks. |
| `paper table/tab_delta_AGM.tex` | Final paper table of AGM-RM differences. |
| `paper table/tab_objdiscov_absolute_AGM.tex` | Final paper table of absolute object-discovery results. |
| `ckpt/checkpoint notes` | Checkpoint directory layout and naming instructions. Checkpoint files are intentionally omitted. |

## Typical commands

Run from the repository root:

```bash
python "add experiments/tables_AGM.py" --check_paper
python "add experiments/plot_seed_effects_AGM.py"
python "add experiments/plot_training_curves_AGM.py"
python "add experiments/plot_main_qualitative_AGM.py"
```

The benchmark and actual-mask scripts require CUDA, datasets, checkpoints, and their corresponding logs. Generated files are written below `add experiments/outputs/` when those scripts are run locally; generated outputs are not included in the release package.
