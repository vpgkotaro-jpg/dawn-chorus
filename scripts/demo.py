"""Live demo: one qubit, 16 circuits. Uniform delays leave several answers, the chirp leaves one.

  python scripts/demo.py --backend fake_sherbrooke --delta 180
"""
import argparse
import pathlib
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dawnchorus import device, estimator, schedules as sc

warnings.filterwarnings("ignore")
TAU_MAX, F0, SHOTS, M = 40e-6, 400e3, 256, 16
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake_sherbrooke")
    ap.add_argument("--qubit", type=int)
    ap.add_argument("--delta", type=float, default=180.0, help="injected detuning in kHz")
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    b = FakeProviderForBackendV2().backend(a.backend)
    good = [q for q in device.sample_qubits(rng, per_backend=b.num_qubits) if q.backend == a.backend]
    q = next(x for x in good if x.index == a.qubit) if a.qubit is not None else min(good, key=lambda x: abs(x.t2 - 80e-6))
    delta = a.delta * 1e3
    sampler = device.full_stack_sampler(device.full_stack_noise(b), seed=7)
    print(f"{a.backend} qubit {q.index}: reported T2 {q.t2 * 1e6:.1f} us, injected detuning {a.delta:.1f} kHz")
    print(f"{M} circuits x {SHOTS} shots = {M * SHOTS} shots\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True, facecolor="#fcfcfb")
    for ax, kind, taus, color in ((axes[0], "uniform", sc.uniform(M, TAU_MAX), ORANGE),
                                  (axes[1], "chirp", sc.chirp(M, TAU_MAX), BLUE)):
        y, circuits = device.full_stack_scan(b, sampler, q.index, taus, delta, F0, SHOTS)
        d_hat, t2_hat = estimator.fit(taus, y, F0)
        answers = estimator.candidates(taus, y, F0)
        listed = ", ".join(f"{d / 1e3:.1f}" for d, _ in answers)
        print(f"{kind:8s} {len(answers)} answer(s) fit the data: {listed} kHz")
        if len(answers) == 1:
            print(f"{'':8s} refined: {d_hat / 1e3:.2f} kHz (error {(d_hat - delta) / 1e3:+.2f} kHz), T2 {t2_hat * 1e6:.1f} us")
        fine = np.linspace(0, TAU_MAX, 1500)
        ax.plot(fine * 1e6, estimator.signal(fine, delta, q.t2, F0), color=GRID, lw=1, zorder=1)
        ax.scatter(taus * 1e6, y, s=36, color=color, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
        ax.set_title(f"{kind}: {len(answers)} answer{'s' if len(answers) > 1 else ''} fit", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("delay (µs)", color=MUTED)
        ax.tick_params(colors=MUTED, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_facecolor("#fcfcfb")
    axes[0].set_ylabel("excited fraction", color=MUTED)
    ops = {k: v for k, v in circuits[-1].count_ops().items() if k != "delay"}
    print("\nNative gates in the longest chirp circuit:", ops)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "demo.png", dpi=160)


if __name__ == "__main__":
    main()
