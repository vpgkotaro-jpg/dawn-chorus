"""Draw the README figures from results/benchmark.csv."""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dawnchorus import schedules as sc

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
COLOR = {"chirp": "#2a78d6", "uniform_nyquist": "#eb6834", "geometric": "#1baf7a", "random": "#eda100", "two_stage": "#e87ba4"}
LABEL = {"chirp": "chirp", "uniform_nyquist": "uniform, alias-free", "geometric": "geometric", "random": "random", "two_stage": "two-stage"}
MARK = {"chirp": "o", "uniform_nyquist": "s", "geometric": "^", "random": "D", "two_stage": "v"}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def ambiguity():
    d = np.linspace(-800e3, 800e3, 6401)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, facecolor=SURFACE)
    for ax, name, t, c in ((axes[0], "Uniform", sc.uniform(16, 40e-6), "#eb6834"),
                           (axes[1], "Chirp", sc.chirp(16, 40e-6), "#2a78d6")):
        a = np.abs(np.exp(2j * np.pi * np.outer(d, t)).mean(axis=1))
        side = sc.ambiguity(t)[1]
        ax.plot(d / 1e3, a, color=c, lw=1.2)
        ax.set_title(f"{name}: peak sidelobe {side:.2f}", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("detuning difference (kHz)", color=MUTED)
        style(ax)
    axes[0].set_ylabel("similarity of two scans", color=MUTED)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "ambiguity.png", dpi=160)


def benchmark():
    rows = list(csv.DictReader(open(ROOT / "results" / "benchmark.csv")))
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    for k in COLOR:
        r = [x for x in rows if x["schedule"] == k]
        xs, ys = [int(x["circuits"]) for x in r], [float(x["rmse_khz"]) for x in r]
        ax.plot(xs, ys, color=COLOR[k], lw=2, marker=MARK[k], ms=8, mec=SURFACE, mew=2, label=LABEL[k])
    ax.set_xlabel("circuits in the scan (256 shots each)", color=MUTED)
    ax.set_ylabel("detuning RMSE (kHz)", color=MUTED)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, labelcolor=MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "benchmark.png", dpi=160)


ambiguity()
benchmark()
