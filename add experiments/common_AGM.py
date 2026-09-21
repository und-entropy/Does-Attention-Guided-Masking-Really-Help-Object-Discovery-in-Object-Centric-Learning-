"""Shared utilities for the primary evaluation and actual-mask capture.

Purpose: parse configurations and logs, select checkpoints, build datasets/models,
and normalize first-frame results. This module is not a standalone entry point.
It is imported by ``benchmark_AGM.py`` and ``capture_actual_masks_AGM.py``.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch as pt
import torch.nn.functional as F
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from object_centric_bench.datum import DataLoader
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap
from object_centric_bench.util import Config, build_from_config


SEEDS = [42, 43, 44]
METRICS = ["ari", "ari_fg", "mbo", "miou"]
PAPER_METRICS = ["ari", "ari_fg", "mbo", "miou"]
PTH_MIN_BYTES = 10 * 1024 * 1024


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    cfg_name: str
    weight_name: str
    data_name: str
    is_video: bool

    @property
    def is_img(self) -> bool:
        return not self.is_video


DATASETS = [
    DatasetSpec("clevrtex", "CLEVRTex", "smoothsa_r-clevrtex_AGM.py", "smoothsa_r-clevrtex", "clevrtex", False),
    DatasetSpec("coco", "COCO", "smoothsa_r-coco_AGM.py", "smoothsa_r-coco", "coco", False),
    DatasetSpec("voc", "VOC", "smoothsa_r-voc_AGM.py", "smoothsa_r-voc", "voc", False),
    DatasetSpec("movi_c", "MOVi-C", "smoothsav_c-movi_c_AGM.py", "smoothsav_c-movi_c", "movi_c", True),
    DatasetSpec("movi_e", "MOVi-E", "smoothsav_c-movi_e_AGM.py", "smoothsav_c-movi_e", "movi_e", True),
    DatasetSpec("ytvis", "YTVIS-2022", "smoothsav_r-ytvis_AGM.py", "smoothsav_r-ytvis", "ytvis_2022", True),
]
DATASET_BY_KEY = {d.key: d for d in DATASETS}


def _pick_path(args: argparse.Namespace, name: str, default: Path) -> Path:
    value = getattr(args, name, None)
    return Path(default if value is None else value).resolve()


def _pick_existing(args: argparse.Namespace, name: str, defaults: list[Path]) -> Path:
    value = getattr(args, name, None)
    if value is not None:
        return Path(value).resolve()
    for candidate in defaults:
        if candidate.exists():
            return candidate.resolve()
    return defaults[0].resolve()


def project_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = Path(getattr(args, "root", ROOT)).resolve()
    return {
        "root": root,
        "config_root": _pick_path(args, "config_root", root / "config-smoothsa"),
        "pgm_weight_root": _pick_existing(args, "pgm_weight_root", [root / "add experiments/ckpt/AGM"]),
        "rand_weight_root": _pick_existing(
            args,
            "rand_weight_root",
            [root / "add experiments/ckpt/RM"],
        ),
        "data_root": _pick_path(args, "data_root", root.parent),
        "out_root": _pick_path(args, "out_root", root / "add experiments" / "outputs"),
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config_root", type=Path, default=None)
    parser.add_argument("--pgm_weight_root", type=Path, default=None)
    parser.add_argument("--rand_weight_root", type=Path, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--out_root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--datasets", nargs="+", default=[d.key for d in DATASETS], choices=[d.key for d in DATASETS])
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--max_batches", type=int, default=None, help="调试用批次数上限；正式任务优先使用各任务的样本数参数。")


def parse_log_file(log_file: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not log_file.exists():
        return pd.DataFrame()
    for line in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(line)
            except Exception:
                continue
        if not isinstance(obj, dict) or len(obj) != 1:
            continue
        step_key, metrics = next(iter(obj.items()))
        if not isinstance(metrics, dict):
            continue
        if "val" not in str(step_key).lower() or not all(k in metrics for k in METRICS):
            continue
        match = re.search(r"\d+", str(step_key))
        epoch = int(match.group(0)) if match else None
        score = sum(float(metrics[k]) for k in METRICS)
        rows.append({"epoch": epoch, "score": score, **{k: float(metrics[k]) for k in METRICS}})
    return pd.DataFrame(rows)


def choose_best_epoch_from_log(log_file: Path) -> tuple[int | None, float | None, pd.DataFrame]:
    df = parse_log_file(log_file)
    if df.empty:
        return None, None, df
    row = df.loc[df["score"].astype(float).idxmax()]
    return int(row["epoch"]), float(row["score"]), df


def find_log_file(weight_dir: Path, seed: int) -> Path | None:
    seed_dir = weight_dir / f"seed{seed}"
    preferred = [
        weight_dir / f"seed{seed}.txt",
        weight_dir / f"{seed}.txt",
        seed_dir / f"seed{seed}.txt",
        seed_dir / f"{seed}.txt",
    ]
    preferred_resolved = [path.resolve() for path in preferred if path.is_file()]
    candidates = list(preferred_resolved)
    if seed_dir.exists():
        candidates.extend(seed_dir.rglob("*.txt"))
    candidates.extend(weight_dir.glob(f"*{seed}*.txt"))
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        return None

    # W&B 导出目录的日志名不一定固定；优先选可解析验证记录最完整的文件。
    ranked = []
    for path in candidates:
        parsed_rows = len(parse_log_file(path))
        preferred_rank = preferred_resolved.index(path) if path in preferred_resolved else len(preferred)
        ranked.append((parsed_rows, -preferred_rank, path.stat().st_mtime_ns, str(path), path))
    return max(ranked)[-1]


def choose_checkpoint(weight_dir: Path, seed: int) -> dict[str, Any]:
    seed_dir = weight_dir / f"seed{seed}"
    log_file = find_log_file(weight_dir, seed)
    best_epoch, best_score, val_df = choose_best_epoch_from_log(log_file) if log_file is not None else (None, None, pd.DataFrame())

    pth_candidates: list[Path] = []
    if seed_dir.exists():
        pth_candidates.extend(seed_dir.rglob("*.pth"))
    pth_candidates.extend(weight_dir.glob("*.pth"))
    pths = sorted({p.resolve() for p in pth_candidates if p.is_file() and p.stat().st_size >= PTH_MIN_BYTES})

    def belongs_to_seed(path: Path) -> bool:
        return (
            f"seed{seed}" in path.parts
            or path.name.startswith(f"{seed}-")
            or path.name.startswith(f"seed{seed}")
            or path.name == "best.pth"
        )

    def epochs_in_name(path: Path) -> list[int]:
        nums = [int(x) for x in re.findall(r"\d+", path.stem)]
        if (f"seed{seed}" in path.parts or path.name.startswith(f"{seed}-")) and nums and nums[0] == seed:
            return nums[1:]
        return nums

    seed_specific_pths = [
        p for p in pths
        if belongs_to_seed(p) and (p.name != "best.pth" or f"seed{seed}" in p.parts)
    ]
    seed_pths = seed_specific_pths or [p for p in pths if p.name == "best.pth"]
    chosen = None
    reason = "missing"

    # Checkpoints are saved only after ``since_step``.  The global log optimum can
    # therefore predate every retained checkpoint.  Select the best *available*
    # checkpoint with an exactly matching validation epoch; never approximate an
    # unavailable optimum by its nearest filename.
    epoch_to_pths: dict[int, list[Path]] = {}
    for path in seed_pths:
        for epoch in epochs_in_name(path):
            epoch_to_pths.setdefault(epoch, []).append(path)
    if not val_df.empty and epoch_to_pths:
        available = val_df[val_df["epoch"].isin(epoch_to_pths)].copy()
        if not available.empty:
            row = available.loc[available["score"].astype(float).idxmax()]
            best_epoch = int(row["epoch"])
            best_score = float(row["score"])
            chosen = sorted(epoch_to_pths[best_epoch])[-1]
            reason = "best_available_log_epoch"
    if chosen is None and seed_pths:
        chosen = seed_pths[-1]
        epochs = epochs_in_name(chosen)
        best_epoch = epochs[-1] if epochs else None
        best_score = None
        reason = "fallback_unscored_seed_ckpt"
    if chosen is None and (weight_dir / "best.pth").exists():
        chosen = weight_dir / "best.pth"
        reason = "fallback_best_pth"
    return {
        "ckpt": chosen,
        "log_file": log_file,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "reason": reason,
        "num_val_rows": int(len(val_df)),
    }


def cfg_for(paths: dict[str, Path], spec: DatasetSpec) -> Path:
    return paths["config_root"] / spec.cfg_name


def data_for(paths: dict[str, Path], spec: DatasetSpec) -> Path:
    return paths["data_root"] / spec.data_name


def weight_for(paths: dict[str, Path], spec: DatasetSpec, strategy: str) -> Path:
    root = paths["pgm_weight_root"] if strategy == "pgm" else paths["rand_weight_root"]
    return root / spec.weight_name


def move_to_cuda(value: Any) -> Any:
    if isinstance(value, pt.Tensor):
        return value.cuda(non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_cuda(v) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_cuda(v) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_cuda(v) for v in value)
    return value


def move_batch_to_cuda(batch: dict[str, Any]) -> dict[str, Any]:
    for k, v in list(batch.items()):
        batch[k] = move_to_cuda(v)
    return batch


def seed_evaluation(seed: int) -> None:
    """Seed all RNGs used by validation and stochastic Slot initialization."""
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    pt.manual_seed(int(seed))
    if pt.cuda.is_available():
        pt.cuda.manual_seed_all(int(seed))


def build_eval_components(
    cfg_path: Path,
    data_dir: Path,
    batch_size: int | None = None,
    num_workers: int | None = None,
    sample_limit: int | None = None,
):
    cfg = Config.fromfile(cfg_path)
    cfg.name = cfg_path.stem
    cfg.dataset_t.base_dir = cfg.dataset_v.base_dir = data_dir
    if batch_size is not None:
        cfg.batch_size_v = batch_size
    if num_workers is not None:
        cfg.num_work = num_workers
    dataset_v = build_from_config(cfg.dataset_v)
    if sample_limit is not None:
        if sample_limit <= 0:
            raise ValueError(f"sample_limit must be positive, got {sample_limit}")
        count = min(int(sample_limit), len(dataset_v))
        # Evenly cover the validation set instead of taking a potentially ordered prefix.
        indices = np.linspace(0, len(dataset_v) - 1, count).round().astype(int).tolist()
        dataset_v = Subset(dataset_v, indices)
    loader = DataLoader(
        dataset_v,
        cfg.batch_size_v,
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )
    loss_fn = MetricWrap(**build_from_config(cfg.loss_fn_v))
    acc_fn = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v))
    cfg.callback_v = [_ for _ in cfg.callback_v if _.type.__name__ != "SaveModel"]
    for cb in cfg.callback_v:
        if cb.type.__name__ in ["AverageLog", "HandleLog"]:
            cb.log_file = None
    callbacks = build_from_config(cfg.callback_v)
    return cfg, dataset_v, loader, loss_fn, acc_fn, callbacks


def build_model(cfg: Config, ckpt: Path | None):
    raw = build_from_config(cfg.model)
    model = ModelWrap(raw, cfg.model_imap, cfg.model_omap)
    if ckpt is not None:
        model.load(ckpt, None, verbose=False)
    if hasattr(cfg, "freez") and cfg.freez:
        model.freez(cfg.freez, verbose=False)
    return model.cuda().eval()


def first_frame_segment(segment: pt.Tensor) -> pt.Tensor:
    """Return a ``[B, H, W]`` index map from repository segmentation batches.

    Image datasets collate masks as ``[B, H, W, S]`` and video datasets as
    ``[B, T, H, W, S]``.  Older evaluation code treated the final slot axis as
    time, which silently produced ``[B, W, S]`` slices.  Integer index maps are
    also accepted for small standalone tests and downstream utilities.
    """
    if not isinstance(segment, pt.Tensor):
        raise TypeError(f"segment must be a tensor, got {type(segment)!r}")

    if segment.dim() == 5:
        # Repository video mask format: [B, T, H, W, S].
        segment = segment[:, 0]
        segment = segment.to(pt.int64).argmax(dim=-1)
    elif segment.dim() == 4:
        if segment.dtype == pt.bool or segment.dtype.is_floating_point:
            # Repository image mask format: [B, H, W, S].
            segment = segment.to(pt.int64).argmax(dim=-1)
        else:
            # Optional integer video index format: [B, T, H, W].
            segment = segment[:, 0]
    elif segment.dim() != 3:
        raise RuntimeError(
            "Unsupported segment shape; expected [B,H,W,S], [B,T,H,W,S], "
            f"or an integer index map, got {tuple(segment.shape)}"
        )

    if segment.dim() != 3:
        raise RuntimeError(f"Failed to normalize segment shape: {tuple(segment.shape)}")
    return segment.long()


def first_frame_attention(att: pt.Tensor) -> pt.Tensor:
    if att.dim() == 5:
        return att[:, 0]
    if att.dim() == 4:
        return att
    raise RuntimeError(f"Unsupported attention shape: {tuple(att.shape)}")

