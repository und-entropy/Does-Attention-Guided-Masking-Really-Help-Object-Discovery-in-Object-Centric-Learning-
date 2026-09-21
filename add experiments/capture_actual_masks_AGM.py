"""Capture RM/AGM segmentations and actual decoder masks at a fixed budget.

Input: datasets, checkpoints, and the corresponding training logs; CUDA is required.
Output: segmentations, mask arrays, and a sample manifest for the mask-plotting script.
Scope: the paper uses a 50% mask ratio for both methods, so spatial selection is
compared without mixing in different random keep counts.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch as pt
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parent
sys.path.insert(0, str(DEFAULT_ROOT))
sys.path.insert(0, str(HERE))

from common_AGM import (  # noqa: E402
    DATASET_BY_KEY,
    add_common_args,
    build_eval_components,
    build_model,
    cfg_for,
    choose_checkpoint,
    data_for,
    first_frame_attention,
    first_frame_segment,
    move_batch_to_cuda,
    project_paths,
    seed_evaluation,
    weight_for,
)


SAMPLES = {
    "clevrtex": [129, 507],
    "coco": [457, 297],
    "voc": [365, 179],
    "movi_c": [223, 103],
    "movi_e": [127, 169],
    "ytvis": [38, 235],
}


def denormalize_image(batch: dict, cfg) -> np.ndarray:
    image = batch["image"][0] if "image" in batch else batch["video"][0, 0]
    mean = pt.tensor(cfg.IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = pt.tensor(cfg.IMAGENET_STD, device=image.device).view(3, 1, 1)
    image = (image * std + mean).clamp(0, 255).byte()
    return np.ascontiguousarray(image.permute(1, 2, 0).detach().cpu().numpy())


def as_first_token_vector(value: pt.Tensor | np.ndarray, name: str) -> np.ndarray:
    array = value.detach().cpu().numpy() if isinstance(value, pt.Tensor) else np.asarray(value)
    while array.ndim > 1:
        array = array[0]
    if array.ndim != 1:
        raise RuntimeError(f"{name} did not reduce to one token vector: {array.shape}")
    return array


def controlled_mask(
    token_score: np.ndarray,
    mask_ratio: float,
) -> tuple[np.ndarray, int, int]:
    token_count = int(token_score.size)
    side = int(round(math.sqrt(token_count)))
    if side * side != token_count:
        raise RuntimeError(f"Token count is not a square grid: {token_count}")
    mask_count = int(round(token_count * mask_ratio))
    mask_count = min(max(mask_count, 1), token_count - 1)
    visible_count = token_count - mask_count
    order = np.argsort(token_score, kind="stable")
    mask = np.ones(token_count, dtype=bool)
    mask[order[:visible_count]] = False
    if int(mask.sum()) != mask_count:
        raise RuntimeError("Controlled mask count does not match the requested budget")
    return mask, visible_count, mask_count


@contextmanager
def force_decoder_mask_ratio(decoder, mask_ratio: float):
    """Force only the decoder keep-count draw while preserving its token order."""
    original_randint = pt.randint
    original_rand = pt.rand

    def fixed_randint(*args, **kwargs):
        if len(args) >= 3:
            low, high, size = args[:3]
        elif len(args) == 2:
            low, high, size = 0, args[0], args[1]
        else:
            return original_randint(*args, **kwargs)
        shape = tuple(int(value) for value in size)
        if int(low) == 0 and len(shape) == 2 and shape[-1] == 1:
            token_count = int(high) + 1
            mask_count = int(round(token_count * mask_ratio))
            mask_count = min(max(mask_count, 1), token_count - 1)
            visible_count = token_count - mask_count
            dtype = kwargs.get("dtype") or pt.int64
            return pt.full(
                shape,
                visible_count,
                dtype=dtype,
                device=kwargs.get("device"),
            )
        return original_randint(*args, **kwargs)

    def fixed_rand(*size, **kwargs):
        value = original_rand(*size, **kwargs)
        if value.ndim == 2 and value.shape[-1] == 1:
            return pt.zeros_like(value)
        return value

    def activate(_module, _args):
        pt.randint = fixed_randint
        pt.rand = fixed_rand

    def restore(_module, _args, output):
        pt.randint = original_randint
        pt.rand = original_rand
        return output

    pre_handle = decoder.register_forward_pre_hook(activate)
    post_handle = decoder.register_forward_hook(restore)
    try:
        yield
    finally:
        pt.randint = original_randint
        pt.rand = original_rand
        pre_handle.remove()
        post_handle.remove()


def decoder_from_model(model):
    raw_model = getattr(model, "m", None)
    decoder = getattr(raw_model, "decode", None)
    if decoder is None:
        raise RuntimeError("Expected model.m.decode, but the decoder was not found")
    required = {"ranking_mode", "capture_training_vis"}
    missing = [name for name in required if not hasattr(decoder, name)]
    if missing:
        raise RuntimeError(
            "The decoder does not expose actual-mask capture fields: " + ", ".join(missing)
        )
    return decoder


@pt.inference_mode()
def evaluate_and_capture(
    model,
    batch: dict,
    cfg,
    strategy: str,
    random_seed: int,
    mask_ratio: float,
) -> dict[str, np.ndarray | float | int]:
    model.eval()
    seed_evaluation(random_seed)
    with pt.autocast("cuda", enabled=pt.cuda.is_available()):
        output = model(batch=batch, isval=True, model=model)

    image = denormalize_image(batch, cfg)
    gt = first_frame_segment(batch["segment"])[0]
    decoded = first_frame_attention(output.get("attentd", output["attenta"])).float()
    decoded = F.interpolate(decoded, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    prediction = decoded.argmax(dim=1)[0]

    decoder = decoder_from_model(model)
    previous_ranking = decoder.ranking_mode
    previous_capture = decoder.capture_training_vis
    decoder.ranking_mode = "random" if strategy == "rand" else "guided"
    decoder.capture_training_vis = True
    decoder._vis_cache = {}

    try:
        model.eval()
        decoder.train()
        seed_evaluation(random_seed)
        with force_decoder_mask_ratio(decoder, mask_ratio):
            with pt.autocast("cuda", enabled=pt.cuda.is_available()):
                model(batch=batch, isval=True, model=model)
        cache = dict(decoder._vis_cache)
    finally:
        decoder.ranking_mode = previous_ranking
        decoder.capture_training_vis = previous_capture
        model.eval()

    if "mae_mask" not in cache or "token_score" not in cache:
        raise RuntimeError(
            "Decoder training forward did not expose mae_mask/token_score. "
            "Use the current object_centric_bench/model/dias.py."
        )
    actual = as_first_token_vector(cache["mae_mask"], "mae_mask").astype(bool)
    score = as_first_token_vector(cache["token_score"], "token_score").astype(np.float32)
    if actual.size != score.size:
        raise RuntimeError(f"Mask/score token mismatch: {actual.size} versus {score.size}")
    expected, visible_count, mask_count = controlled_mask(score, mask_ratio)
    if not np.array_equal(actual, expected):
        raise RuntimeError(
            "Captured decoder mask does not match the fixed-budget ranking reconstruction"
        )

    return {
        "image": image,
        "gt_index": gt.detach().cpu().numpy().astype(np.int32),
        "pred_index": prediction.detach().cpu().numpy().astype(np.int32),
        "mae_mask_actual": actual.astype(np.uint8),
        "token_score": score,
        "actual_mask_ratio": float(actual.mean()),
        "target_mask_ratio": float(mask_ratio),
        "visible_count": visible_count,
        "mask_count": mask_count,
        "token_count": int(score.size),
    }


def validate_pairs(
    data_dir: Path,
    datasets: list[str],
    seed: int,
    mask_ratio: float,
) -> None:
    for dataset in datasets:
        for sample_index in SAMPLES[dataset]:
            paths = {
                strategy: data_dir / f"{dataset}_{strategy}_seed{seed}_sample{sample_index}.npz"
                for strategy in ("rand", "pgm")
            }
            with np.load(paths["rand"], allow_pickle=False) as rm, np.load(
                paths["pgm"], allow_pickle=False
            ) as agm:
                if not np.array_equal(rm["image"], agm["image"]):
                    raise RuntimeError(f"RM/AGM input mismatch: {dataset}/{sample_index}")
                if not np.array_equal(rm["gt_index"], agm["gt_index"]):
                    raise RuntimeError(f"RM/AGM ground-truth mismatch: {dataset}/{sample_index}")
                for item, name in ((rm, "RM"), (agm, "AGM")):
                    token_count = int(item["token_count"])
                    if item["mae_mask_actual"].size != token_count:
                        raise RuntimeError(f"{name} actual mask is incomplete: {dataset}/{sample_index}")
                    expected_count = int(round(token_count * mask_ratio))
                    actual_count = int(item["mae_mask_actual"].sum())
                    if actual_count != expected_count:
                        raise RuntimeError(
                            f"{name} mask budget is {actual_count}/{token_count}, "
                            f"expected {expected_count}/{token_count}: {dataset}/{sample_index}"
                        )
                rm_count = int(rm["mae_mask_actual"].sum())
                agm_count = int(agm["mae_mask_actual"].sum())
                if rm_count != agm_count:
                    raise RuntimeError(
                        f"RM/AGM actual mask counts differ: "
                        f"{dataset}/{sample_index}: {rm_count} versus {agm_count}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(root=DEFAULT_ROOT, seeds=[42], batch_size=1, num_workers=2)
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.50,
        help=(
            "Shared mask ratio forced inside the captured decoder forward for RM and AGM."
        ),
    )
    parser.add_argument("--mask_seed", type=int, default=20260909)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.seeds != [42]:
        raise ValueError("The registered qualitative figures use seed 42; pass --seeds 42")
    if not math.isclose(args.mask_ratio, 0.50, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The 0909 visualization protocol fixes mask_ratio at 0.50")
    if not pt.cuda.is_available():
        raise RuntimeError("CUDA is required to capture checkpoint-derived masks")

    paths = project_paths(args)
    out_root = Path(args.out_root or (HERE / "outputs/mask_capture")).resolve()
    data_dir = out_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for dataset_offset, dataset_key in enumerate(args.datasets):
        spec = DATASET_BY_KEY[dataset_key]
        targets = SAMPLES[dataset_key]
        cfg, dataset, full_loader, _, _, _ = build_eval_components(
            cfg_for(paths, spec),
            data_for(paths, spec),
            batch_size=1,
            num_workers=args.num_workers,
        )
        subset = Subset(dataset, targets)
        loader = DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=full_loader.collate_fn,
            pin_memory=True,
        )
        del full_loader

        for strategy in ("rand", "pgm"):
            selected = choose_checkpoint(weight_for(paths, spec, strategy), 42)
            checkpoint = selected.get("ckpt")
            if checkpoint is None:
                raise FileNotFoundError(f"No checkpoint found for {dataset_key}/{strategy}/seed42")
            print(
                f"[load] {dataset_key}/{strategy}/seed42: {checkpoint} "
                f"({selected.get('reason', 'unknown')})",
                flush=True,
            )
            model = build_model(cfg, checkpoint)
            captured = 0
            for local_order, batch in enumerate(loader):
                sample_index = targets[local_order]
                destination = data_dir / (
                    f"{dataset_key}_{strategy}_seed42_sample{sample_index}.npz"
                )
                if destination.exists() and not args.overwrite:
                    with np.load(destination, allow_pickle=False) as cached:
                        actual_ratio = float(cached["actual_mask_ratio"])
                        token_count = int(cached["token_count"])
                    print(f"[resume] {destination.name}", flush=True)
                else:
                    batch = move_batch_to_cuda(batch)
                    random_seed = args.mask_seed + dataset_offset * 10_000 + sample_index
                    result = evaluate_and_capture(
                        model,
                        batch,
                        cfg,
                        strategy,
                        random_seed,
                        args.mask_ratio,
                    )
                    np.savez_compressed(
                        destination,
                        **result,
                        dataset=np.asarray(dataset_key),
                        strategy=np.asarray(strategy),
                        display_strategy=np.asarray("RM" if strategy == "rand" else "AGM"),
                        seed=np.asarray(42, dtype=np.int32),
                        sample_index=np.asarray(sample_index, dtype=np.int32),
                        checkpoint=np.asarray(str(checkpoint)),
                        selection_reason=np.asarray(str(selected.get("reason", ""))),
                        mask_seed=np.asarray(random_seed, dtype=np.int64),
                    )
                    actual_ratio = float(result["actual_mask_ratio"])
                    token_count = int(result["token_count"])
                    print(
                        f"[capture] {destination.name}: actual={actual_ratio:.3f}",
                        flush=True,
                    )
                manifest_rows.append(
                    {
                        "dataset": dataset_key,
                        "strategy": strategy,
                        "display_strategy": "RM" if strategy == "rand" else "AGM",
                        "seed": 42,
                        "sample_index": sample_index,
                        "checkpoint": str(checkpoint),
                        "selection_reason": selected.get("reason", ""),
                        "actual_mask_ratio": actual_ratio,
                        "token_count": token_count,
                        "file": str(destination),
                    }
                )
                captured += 1
            if captured != len(targets):
                raise RuntimeError(
                    f"Captured {captured}/{len(targets)} samples for {dataset_key}/{strategy}"
                )
            del model
            pt.cuda.empty_cache()
            gc.collect()
        del loader, subset, dataset, cfg
        gc.collect()

    validate_pairs(data_dir, args.datasets, 42, args.mask_ratio)
    manifest = pd.DataFrame(manifest_rows)
    expected_rows = len(args.datasets) * 2 * 2
    if len(manifest) != expected_rows or manifest.duplicated(
        ["dataset", "strategy", "seed", "sample_index"]
    ).any():
        raise RuntimeError(f"Manifest is incomplete: expected {expected_rows}, got {len(manifest)}")
    manifest.to_csv(out_root / "actual_mask_manifest.csv", index=False, encoding="utf-8-sig")
    (out_root / "capture_protocol.json").write_text(
        json.dumps(
            {
                "target_mask_ratio": args.mask_ratio,
                "mask_seed": args.mask_seed,
                "seed": 42,
                "datasets": args.datasets,
                "samples": {key: SAMPLES[key] for key in args.datasets},
                "mask_source": "decoder._vis_cache['mae_mask'] and token_score",
                "display_rule": "actual decoder mask with a fixed shared keep count",
                "paper_mask_key": "mae_mask_actual",
                "keep_count_forced_during_forward": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] Actual patch masks: {out_root}", flush=True)


if __name__ == "__main__":
    main()
