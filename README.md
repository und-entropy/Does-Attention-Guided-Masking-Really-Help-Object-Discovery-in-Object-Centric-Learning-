# AGM for SmoothSA

Official implementation package for the workshop paper [Does Attention-Guided Masking Really Help Object Discovery in Object-Centric Learning?](https://arxiv.org/abs/2609.15187).

Authors: Youliang Tao, Yanhua Han, Bin Zhao, Juho Kannala, Joni Pajarinen, and Rongzhen Zhao.

This repository is based on [SmoothSA](https://github.com/Genera1Z/SmoothSA). It studies the effect of replacing random masking (RM) in the decoder with attention-guided masking (AGM).

## Method and main finding

AGM ranks image patches using aggregated slot attention, the entropy residual of the final decoder cross-attention, and a small random component. Low-score patches are kept and high-score patches are masked. AGM introduces no learnable parameters; training adds one gradient-free unmasked decoder pass, while inference uses all patches.

The core algorithm is implemented in `object_centric_bench/model/dias_AGM.py`. `smoothsa_AGM.py` is a video interface adapter: it forwards the existing aggregated attention to the AGM decoder without changing the video model structure, temporal propagation, slot iterations, or parameter count. The `_AGM` suffix is retained because the decoder interface differs from upstream SmoothSA; this adapter is not a separate methodological contribution.

The study evaluates six datasets, two masking strategies, and three seeds (42, 43, and 44), for 36 models in total. AGM improves some overall segmentation metrics on COCO and VOC, while foreground object discovery is usually unchanged or lower.

| Dataset | ΔARI | ΔFG-ARI | ΔmBO | ΔmIoU |
|---|---:|---:|---:|---:|
| CLEVRTex | -3.96 | -6.23 | -4.38 | -5.40 |
| COCO | +14.77 | -2.34 | +4.11 | +4.07 |
| VOC | +17.36 | -4.01 | +5.45 | +5.55 |
| YTVIS-2022 | -2.27 | +0.64 | -1.97 | -1.60 |
| MOVi-C | -6.22 | +0.50 | -3.12 | -2.77 |
| MOVi-E | +14.11 | -1.66 | +1.60 | +1.76 |

Values are AGM minus RM mean differences in percentage points. Complete means, sample standard deviations, and per-seed results are generated from the primary comparison data in [`add experiments`](add%20experiments/readme.md).

## Repository layout

```text
SmoothSA_AGM/
├── config-smoothsa/              Six dataset-specific *_AGM.py configurations
├── object_centric_bench/         SmoothSA framework
│   ├── datum/                    Data loading and preprocessing
│   ├── learn/                    Losses, metrics, and training callbacks
│   └── model/
│       ├── dias_AGM.py           AGM decoder and ranking algorithm
│       ├── smoothsa_AGM.py       Video decoder-interface adapter
│       ├── basic_AGM.py          Local DINOv2 checkpoint loading
│       └── other upstream files  Unmodified SmoothSA dependencies
├── res/                          Final workshop figures in PDF format
├── add experiments/              Primary comparison and paper figures/tables
├── train_AGM.py                  Training entry point
├── eval.py                       Unmodified upstream single-model evaluator
├── requirements_AGM.txt          Runtime dependencies
├── readme.md                     This English overview
├── file inventory (Chinese filename)  Detailed inventory and rename mapping
└── LICENSE                       Upstream SmoothSA license
```

Modified implementations and configurations use the `_AGM` suffix before the file extension. Upstream modules needed by package initialization and class dependencies remain under their original names. AGM configurations explicitly import the suffixed implementations.

## Environment and data

Use Python 3.11 with a CUDA-compatible installation of PyTorch and torchvision. Install PyTorch and torchvision for the target CUDA version first, then install the remaining dependencies:

```bash
pip install -r requirements_AGM.txt
```

Tables and statistical plots require NumPy, pandas, and Matplotlib. Rebuilding qualitative figures additionally requires SciPy and PyMuPDF. Dataset loading follows SmoothSA's LMDB format. Expected dataset directory names are `clevrtex`, `coco`, `voc`, `movi_c`, `movi_e`, and `ytvis_2022`.

Model weights are intentionally omitted. Empty checkpoint locations and naming conventions are described in the checkpoint notes file under `add experiments/ckpt/`. The local DINOv2 checkpoint should be copied to `add experiments/ckpt/DINOv2/model.safetensors`. Run all commands from the repository root.

## Training and evaluation

The following example trains COCO with AGM and RM using the same configuration and different ranking modes:

```bash
python train_AGM.py --cfg_file config-smoothsa/smoothsa_r-coco_AGM.py --data_dir /path/to/datasets/coco --save_dir "add experiments/ckpt/AGM" --ranking_mode guided --seeds 42 43 44
python train_AGM.py --cfg_file config-smoothsa/smoothsa_r-coco_AGM.py --data_dir /path/to/datasets/coco --save_dir "add experiments/ckpt/RM" --ranking_mode random --seeds 42 43 44
python eval.py --cfg_file config-smoothsa/smoothsa_r-coco_AGM.py --data_dir /path/to/datasets/coco --ckpt_file /path/to/checkpoint.pth --is_img True
```

`train_AGM.py` creates per-model seed directories and training logs. Video evaluation should omit `--is_img`; the upstream argument parser is not designed for `--is_img False`.

Image experiments use a learning rate of 2e-4 and video experiments use 5e-5. COCO uses 100k steps; the other domains use 50k steps. MOVi-C and MOVi-E initialize slots from first-frame target boxes; the other domains use unconditional initialization.

## License and citation

The base implementation comes from [Genera1Z/SmoothSA](https://github.com/Genera1Z/SmoothSA), whose MIT license is retained. Please cite the workshop paper and SmoothSA together. Dataset and figure copyrights remain with their respective authors.
