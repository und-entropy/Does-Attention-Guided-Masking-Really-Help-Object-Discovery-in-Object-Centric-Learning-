"""Build six-column RM/AGM segmentation and actual-mask figures.

Input: arrays and the manifest produced by the mask-capture script; both methods
must use a 50% mask ratio. Output: image- and video-domain qualitative figures.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
RM = "#287E8A"
AGM = "#C2773E"
INK = "#253239"
DATASETS = ["clevrtex", "coco", "voc", "movi_c", "movi_e", "ytvis"]
LABELS = {
    "clevrtex": "CLEVRTex",
    "coco": "COCO",
    "voc": "VOC",
    "movi_c": "MOVi-C",
    "movi_e": "MOVi-E",
    "ytvis": "YTVIS-2022",
}
SAMPLES = {
    "clevrtex": [129, 507],
    "coco": [457, 297],
    "voc": [365, 179],
    "movi_c": [223, 103],
    "movi_e": [127, 169],
    "ytvis": [38, 235],
}


def theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def palette(count: int) -> np.ndarray:
    colors = [
        "#BDC7CD", "#DFAD4D", "#5295C2", "#A981B5", "#62AA87", "#D47778",
        "#729DD5", "#C79469", "#8DAF4F", "#A0A3DA", "#48A9A6", "#D783AF",
        "#9EBDCF", "#D6A485", "#99B78C", "#A584B0", "#9C8E63", "#C1716C",
        "#608BBA", "#DAAFB7", "#74948D", "#959CDA", "#88B9B0", "#BAAA6E",
    ]
    base = [to_rgb(color) for color in colors]
    return np.asarray(
        [
            base[index]
            if index < len(base)
            else matplotlib.colors.hsv_to_rgb([(index * 0.61803398875) % 1, 0.40, 0.84])
            for index in range(count)
        ]
    )


def render_segmentation(image: np.ndarray, labels: np.ndarray, colors: dict[int, np.ndarray]) -> np.ndarray:
    rgb = np.empty_like(image, dtype=float)
    for label, color in colors.items():
        rgb[labels == label] = color
    result = 0.24 * image.astype(float) / 255 + 0.76 * rgb
    edge = np.zeros(labels.shape, dtype=bool)
    edge[1:] |= labels[1:] != labels[:-1]
    edge[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    result[edge] = 0.20 * result[edge] + 0.80
    return np.uint8(np.clip(result, 0, 1) * 255)


def matched_colors(pred: np.ndarray, gt: np.ndarray, gt_colors: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    pred_labels = np.unique(pred)
    gt_labels = np.unique(gt)
    intersection = np.asarray(
        [[np.logical_and(pred == p, gt == g).sum() for g in gt_labels] for p in pred_labels],
        dtype=float,
    )
    union = (
        np.asarray([(pred == p).sum() for p in pred_labels])[:, None]
        + np.asarray([(gt == g).sum() for g in gt_labels])[None, :]
        - intersection
    )
    rows, columns = linear_sum_assignment(-intersection / np.maximum(union, 1))
    result = {
        int(pred_labels[row]): gt_colors[int(gt_labels[column])]
        for row, column in zip(rows, columns)
    }
    unmatched = [int(label) for label in pred_labels if int(label) not in result]
    remaining = palette(len(gt_colors) + len(unmatched))[len(gt_colors):]
    result.update(dict(zip(unmatched, remaining)))
    return result


def mask_overlay(image: np.ndarray, mask_1d: np.ndarray, color: str) -> np.ndarray:
    mask_1d = np.asarray(mask_1d).astype(bool)
    side = int(round(math.sqrt(mask_1d.size)))
    if side * side != mask_1d.size:
        raise ValueError(f"Mask is not a square patch grid: {mask_1d.size}")
    patch_mask = mask_1d.reshape(side, side)
    row_index = np.minimum(np.arange(image.shape[0]) * side // image.shape[0], side - 1)
    col_index = np.minimum(np.arange(image.shape[1]) * side // image.shape[1], side - 1)
    pixel_mask = patch_mask[row_index[:, None], col_index[None, :]]
    output = image.astype(float).copy()
    neutral = np.asarray([239, 241, 242], dtype=float)
    output[pixel_mask] = 0.15 * output[pixel_mask] + 0.85 * neutral

    boundary = np.zeros(pixel_mask.shape, dtype=bool)
    boundary[1:] |= pixel_mask[1:] != pixel_mask[:-1]
    boundary[:, 1:] |= pixel_mask[:, 1:] != pixel_mask[:, :-1]
    edge_color = np.asarray(to_rgb(color)) * 255
    output[boundary] = edge_color

    patch_h = image.shape[0] / side
    patch_w = image.shape[1] / side
    grid_color = np.asarray([205, 211, 214], dtype=float)
    for index in range(1, side):
        y = min(int(round(index * patch_h)), image.shape[0] - 1)
        x = min(int(round(index * patch_w)), image.shape[1] - 1)
        output[y : y + 1] = 0.65 * output[y : y + 1] + 0.35 * grid_color
        output[:, x : x + 1] = 0.65 * output[:, x : x + 1] + 0.35 * grid_color
    return np.uint8(np.clip(output, 0, 255))


def load_pair(
    data_dir: Path,
    dataset: str,
    sample_index: int,
    mask_key: str,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    paths = {
        strategy: data_dir / f"{dataset}_{strategy}_seed42_sample{sample_index}.npz"
        for strategy in ("rand", "pgm")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    with np.load(paths["rand"], allow_pickle=False) as rm, np.load(
        paths["pgm"], allow_pickle=False
    ) as agm:
        image = rm["image"].copy()
        gt = rm["gt_index"].copy()
        if not np.array_equal(image, agm["image"]):
            raise ValueError(f"RM/AGM input mismatch: {dataset}/{sample_index}")
        if not np.array_equal(gt, agm["gt_index"]):
            raise ValueError(f"RM/AGM ground-truth mismatch: {dataset}/{sample_index}")
        gt_labels = np.unique(gt)
        gt_colors = dict(zip(gt_labels.tolist(), palette(len(gt_labels))))
        rm_ratio = float(np.asarray(rm[mask_key]).mean())
        agm_ratio = float(np.asarray(agm[mask_key]).mean())
        panels = [
            image,
            render_segmentation(image, gt, gt_colors),
            render_segmentation(
                image,
                rm["pred_index"],
                matched_colors(rm["pred_index"], gt, gt_colors),
            ),
            mask_overlay(image, rm[mask_key], RM),
            render_segmentation(
                image,
                agm["pred_index"],
                matched_colors(agm["pred_index"], gt, gt_colors),
            ),
            mask_overlay(image, agm[mask_key], AGM),
        ]
        rm_count = int(np.asarray(rm[mask_key]).sum())
        agm_count = int(np.asarray(agm[mask_key]).sum())
        if rm_count != agm_count or not np.isclose(rm_ratio, agm_ratio):
            raise ValueError(
                f"RM/AGM mask budgets differ: {dataset}/{sample_index}: "
                f"{rm_count} versus {agm_count} patches"
            )
    return panels, (rm_ratio, agm_ratio)


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{name}.pdf",
        bbox_inches="tight",
        pad_inches=0.025,
        metadata={"Author": "", "Creator": "Matplotlib", "CreationDate": None},
    )
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=360,
        bbox_inches="tight",
        pad_inches=0.025,
    )
    plt.close(fig)


def build_group(
    data_dir: Path,
    output_dir: Path,
    name: str,
    datasets: list[str],
    mask_key: str,
) -> None:
    rows = [(dataset, index) for dataset in datasets for index in SAMPLES[dataset]]
    fig, axes = plt.subplots(6, 6, figsize=(7.25, 7.42))
    fig.subplots_adjust(
        left=0.009,
        right=0.997,
        bottom=0.006,
        top=0.946,
        hspace=0.135,
        wspace=0.022,
    )
    headers = [
        "Input",
        "Ground truth",
        "RM\nsegmentation",
        "RM\nmask",
        "AGM\nsegmentation",
        "AGM\nmask",
    ]
    colors = [INK, INK, RM, RM, AGM, AGM]
    for column, (header, color) in enumerate(zip(headers, colors)):
        axes[0, column].text(
            0.5,
            1.16,
            header,
            transform=axes[0, column].transAxes,
            ha="center",
            va="bottom",
            fontsize=7.6,
            linespacing=0.92,
            fontweight="semibold",
            color=color,
        )

    display_ratio = None
    for row, (dataset, sample_index) in enumerate(rows):
        panels, ratios = load_pair(data_dir, dataset, sample_index, mask_key)
        display_ratio = ratios[0] if display_ratio is None else display_ratio
        if not np.isclose(display_ratio, ratios[0]):
            raise ValueError("All panels must use one shared mask ratio")
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel, interpolation="nearest")
            axes[row, column].set_axis_off()
            if column in (2, 3, 4, 5):
                method_color = RM if column in (2, 3) else AGM
                axes[row, column].add_patch(
                    Rectangle(
                        (0, 0),
                        1,
                        1,
                        transform=axes[row, column].transAxes,
                        fill=False,
                        edgecolor=method_color,
                        linewidth=0.65,
                    )
                )
        axes[row, 0].text(
            0,
            1.025,
            f"{LABELS[dataset]} / {sample_index}",
            transform=axes[row, 0].transAxes,
            ha="left",
            va="bottom",
            fontsize=6.7,
            color=INK,
        )
    note = f"Shared mask ratio: {100 * float(display_ratio):.0f}%"
    fig.text(0.995, 0.003, note, ha="right", va="bottom", fontsize=5.7, color="#6D787E")
    save_figure(fig, output_dir, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture_root", type=Path, default=HERE / "outputs/mask_capture")
    parser.add_argument("--out_root", type=Path, default=HERE / "outputs/mask_figures")
    args = parser.parse_args()
    data_dir = args.capture_root.resolve() / "data"
    output_dir = args.out_root.resolve()
    manifest = args.capture_root.resolve() / "actual_mask_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Run capture_actual_masks_AGM.py first: {manifest}")
    frame = pd.read_csv(manifest)
    if "actual_mask_ratio" not in frame or not np.allclose(
        frame["actual_mask_ratio"].to_numpy(float), 0.50, rtol=0.0, atol=1e-12
    ):
        raise ValueError(
            "The workshop figure requires 50% masks. Re-run capture with --mask_ratio 0.50 --overwrite."
        )
    expected = {
        (dataset, strategy, 42, sample)
        for dataset in DATASETS
        for strategy in ("rand", "pgm")
        for sample in SAMPLES[dataset]
    }
    actual = set(
        frame[["dataset", "strategy", "seed", "sample_index"]].itertuples(
            index=False, name=None
        )
    )
    if actual != expected:
        raise ValueError(f"Actual-mask manifest is incomplete: {len(actual)}/{len(expected)}")
    theme()
    build_group(data_dir, output_dir, "examples_images", DATASETS[:3], "mae_mask_actual")
    build_group(data_dir, output_dir, "examples_videos", DATASETS[3:], "mae_mask_actual")
    print(f"[done] Six-column fixed-budget actual-mask figures: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
