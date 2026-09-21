"""Plot per-seed metric differences between AGM and RM.

Input: the 36 real model-evaluation records in ``data/primary_AGM.csv``; no
model weights are required. Output: four metric panels and paired differences.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from tables_AGM import LABELS, METRICS, HEADERS, validate
DATASETS = ["clevrtex", "coco", "voc", "movi_c", "movi_e", "ytvis"]
HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ASSETS = HERE / "outputs/seed_effects"
RAND, PGM, INK = "#287E8A", "#C2773E", "#253239"
def theme():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "axes.titleweight": "semibold", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#AAB3B7", "axes.linewidth": .6,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.color": "#E5E9EB", "grid.linewidth": .6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
        "svg.fonttype": "none", "lines.solid_capstyle": "round",
    })


def save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / f"{name}.pdf", bbox_inches="tight", pad_inches=.035,
                metadata={"Author": "", "CreationDate": None})
    fig.savefig(ASSETS / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=.035)
    plt.close(fig)


def seed_effects():
    raw = pd.read_csv(DATA / "primary_AGM.csv")
    validate(raw, ["dataset", "strategy", "seed"], 36)
    wide = raw.pivot(index=["dataset", "seed"], columns="strategy", values=METRICS)
    delta = pd.DataFrame({m: 100 * (wide[m]["pgm"] - wide[m]["rand"]) for m in METRICS}).reset_index()
    ASSETS.mkdir(parents=True, exist_ok=True)
    delta.to_csv(ASSETS / "seed_effects_AGM.csv", index=False)
    delta = pd.read_csv(ASSETS / "seed_effects_AGM.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7, 3.9), layout="constrained")
    markers = ["o", "s", "^"]
    for panel, (ax, metric, header) in enumerate(zip(axes.flat, METRICS, HEADERS)):
        ax.axvline(0, color="#6C777C", lw=.8, ls=(0, (3, 3)), zorder=1)
        for j, dataset in enumerate(DATASETS):
            values = delta.loc[delta.dataset == dataset].sort_values("seed")[metric].to_numpy()
            ax.hlines(j, values.min(), values.max(), color="#BEC7CB", lw=1.1, zorder=1)
            for value, offset, marker in zip(values, [-.14, 0, .14], markers):
                ax.scatter(value, j + offset, marker=marker, s=23,
                           facecolor=PGM if value >= 0 else RAND, edgecolor="white", lw=.4, zorder=3)
            ax.scatter(values.mean(), j, marker="|", s=120, color=INK, lw=1.5, zorder=4)
        ax.set_yticks(range(6), [LABELS[d] for d in DATASETS])
        ax.set_ylim(5.55, -.6)
        ax.set_title(f"({chr(97+panel)}) {header}", loc="left", pad=7)
        ax.grid(axis="x", zorder=0)
        ax.margins(x=.12)
        ax.set_xlabel("AGM - RM (percentage points)")
    handles = [Line2D([], [], marker=m, color="#58656C", ls="", markersize=4,
                      label=f"Seed {s}") for m, s in zip(markers, [42, 43, 44])]
    handles.append(Line2D([], [], marker="|", color=INK, ls="", markersize=8, label="Mean"))
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
               handletextpad=.35, columnspacing=1.5, fontsize=8)
    save(fig, "fig_seed_effects_AGM")


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    theme()
    seed_effects()
