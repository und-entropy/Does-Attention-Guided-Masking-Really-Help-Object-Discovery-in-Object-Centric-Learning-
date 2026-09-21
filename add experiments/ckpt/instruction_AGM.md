# Checkpoint placeholders

Weights are intentionally omitted from this package. Copy them into the following layout:

```text
ckpt/
├── DINOv2/model.safetensors
├── RM/<model_name>/seed42/*.pth
│                       seed43/*.pth
│                       seed44/*.pth
└── AGM/<model_name>/seed42/*.pth
                        seed43/*.pth
                        seed44/*.pth
```

The package reserves six model directories for each strategy and three seeds, for 36 model slots in total. `.gitkeep` files only preserve empty directories; they are not checkpoints.

Model names are `smoothsa_r-clevrtex`, `smoothsa_r-coco`, `smoothsa_r-voc`, `smoothsav_c-movi_c`, `smoothsav_c-movi_e`, and `smoothsav_r-ytvis`. Do not append `_AGM` to these directory names, so they remain compatible with historical checkpoints.

Training logs may be stored as `seed42.txt`, `seed43.txt`, and `seed44.txt` under each model directory. The legacy flat layout containing files such as `42-0025.pth` and `42.txt` is also supported. Keep the original seed and epoch names, and do not share one `best.pth` across seeds.

Checkpoint files are ignored by `.gitignore` and should be distributed separately by the authors. Dataset files are not included.
