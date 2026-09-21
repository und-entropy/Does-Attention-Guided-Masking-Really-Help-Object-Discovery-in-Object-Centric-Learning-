"""Plot the paper's RM/AGM training curves from historical validation records.

Input: ``data/training_curves_AGM.csv``; no model weights are required. Output:
ARI and FG-ARI PDF/PNG curves. COCO is shown for 100k steps and other datasets
for the first 50k steps; no metrics are synthesized or interpolated.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/training_curves_AGM.csv"
OUT = ROOT / "outputs/training_curves"

DATASETS = ["clevrtex", "coco", "voc", "movi_c", "movi_e", "ytvis"]
LABELS = {
    "clevrtex": "CLEVRTex",
    "coco": "COCO",
    "voc": "VOC",
    "movi_c": "MOVi-C",
    "movi_e": "MOVi-E",
    "ytvis": "YTVIS-2022",
}
DISPLAY_STEPS = {dataset: (100_000 if dataset == "coco" else 50_000) for dataset in DATASETS}
METRICS = [("ari", "ARI"), ("ari_fg", "FG-ARI")]
COLORS = {"rand": "#287E8A", "pgm": "#C2773E"}
STYLES = {"rand": "-", "pgm": "--"}
INK = "#253239"


def configure_theme() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "axes.labelsize": 8.4,
            "axes.titlesize": 9.0,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#AAB3B7",
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "grid.color": "#E4E9EB",
            "grid.linewidth": 0.6,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def prepare_saved_data():
    plot_data = pd.read_csv(SOURCE)
    audit = plot_data.groupby(["dataset", "strategy", "seed"]).size().reset_index(name="records")
    if len(audit) != 36:
        raise ValueError("Training curves require 36 groups: six datasets, two strategies, and three seeds")
    return plot_data


def draw() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    configure_theme()
    data = prepare_saved_data()
    stats = (
        data.groupby(["dataset", "strategy", "estimated_step_k"], as_index=False)
        .agg(
            ari_mean=("ari", "mean"),
            ari_sd=("ari", "std"),
            ari_fg_mean=("ari_fg", "mean"),
            ari_fg_sd=("ari_fg", "std"),
            seed_count=("seed", "nunique"),
        )
    )
    if not stats.seed_count.eq(3).all():
        raise ValueError("Every plotted point must contain all three seeds")

    fig, axes = plt.subplots(4, 3, figsize=(7.05, 6.12), dpi=300)
    for block, datasets in enumerate((DATASETS[:3], DATASETS[3:])):
        for col, dataset in enumerate(datasets):
            for metric_index, (metric, metric_label) in enumerate(METRICS):
                ax = axes[2 * block + metric_index, col]
                lower, upper = [], []
                for strategy in ("rand", "pgm"):
                    sub = stats.loc[
                        stats.dataset.eq(dataset) & stats.strategy.eq(strategy)
                    ].sort_values("estimated_step_k")
                    x = sub.estimated_step_k.to_numpy()
                    y = sub[f"{metric}_mean"].to_numpy() * 100.0
                    sd = sub[f"{metric}_sd"].to_numpy() * 100.0
                    ax.plot(x, y, color=COLORS[strategy], ls=STYLES[strategy], lw=1.45, zorder=3)
                    ax.fill_between(
                        x,
                        y - sd,
                        y + sd,
                        color=COLORS[strategy],
                        alpha=0.12,
                        linewidth=0,
                        zorder=1,
                    )
                    ax.scatter(x, y, s=5, color=COLORS[strategy], alpha=0.32, linewidth=0, zorder=4)
                    lower.extend(y - sd)
                    upper.extend(y + sd)

                ax.grid(axis="y", zorder=0)
                max_step_k = DISPLAY_STEPS[dataset] / 1000.0
                ax.set_xlim(0, max_step_k + 0.5)
                if dataset == "coco":
                    ax.set_xticks([0, 20, 40, 60, 80, 100])
                else:
                    ax.set_xticks([0, 10, 20, 30, 40, 50])
                ax.set_ylim(max(0, min(lower) - 5), min(100, max(upper) + 5))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.set_ylabel(metric_label)
                ax.set_xlabel("")
                if metric_index == 0:
                    panel = chr(ord("a") + DATASETS.index(dataset))
                    title = f"({panel}) {LABELS[dataset]}"
                    if dataset == "coco":
                        title += " | 100k"
                    ax.set_title(title, loc="left", pad=6)

    fig.legend(
        handles=[
            Line2D([], [], color=COLORS["rand"], lw=1.55, label="RM"),
            Line2D([], [], color=COLORS["pgm"], lw=1.55, ls="--", label="AGM"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        ncol=2,
        fontsize=8.1,
    )
    fig.text(
        0.535,
        0.018,
        "Training steps (thousands; COCO 100k, others 50k)",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=INK,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.065, top=0.93, hspace=0.50, wspace=0.34)

    pdf = OUT / "fig_training_curves_AGM.pdf"
    png = OUT / "fig_training_curves_AGM.png"
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.035, metadata={"Author": "", "CreationDate": None})
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    draw()
