"""Compare delay schedules over qubits from IBM snapshot calibrations. Writes results/."""
import csv
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dawnchorus import device, estimator, schedules as sc

TAU_MAX, F0, SPAN, SHOTS = 40e-6, 400e3, 300e3, 256
FAIL = 10e3
MS = (8, 10, 12, 16, 24)
TRIALS = 300
F_MAX = F0 + 450e3


def make(kind, m, rng):
    return {"uniform": lambda: sc.uniform(m, TAU_MAX),
            "uniform_nyquist": lambda: sc.uniform_nyquist(m, F_MAX),
            "two_stage": lambda: sc.two_stage(m, TAU_MAX, F_MAX),
            "chirp": lambda: sc.chirp(m, TAU_MAX),
            "geometric": lambda: sc.geometric(m, TAU_MAX),
            "random": lambda: sc.random_uniform(m, TAU_MAX, rng)}[kind]()


def main():
    rng = np.random.default_rng(2026)
    qubits = device.sample_qubits(rng)
    print(f"{len(qubits)} qubits from {len({q.backend for q in qubits})} backends", flush=True)
    rows, t0 = [], time.time()
    for m in MS:
        trials = [(qubits[rng.integers(len(qubits))], rng.uniform(-SPAN, SPAN)) for _ in range(TRIALS)]
        for kind in ("uniform", "uniform_nyquist", "two_stage", "geometric", "random", "chirp"):
            err, ratio, bound = [], [], []
            for q, delta in trials:
                taus = make(kind, m, rng)
                p = np.clip(device.fast_scan(q, taus, delta, F0), 0, 1)
                y = rng.binomial(SHOTS, p) / SHOTS
                d_hat, t2_hat = estimator.fit(taus, y, F0)
                err.append(d_hat - delta)
                ratio.append(t2_hat / q.t2)
                bound.append(estimator.crlb(taus, q.t2, SHOTS, F0))
            err, ratio = np.array(err), np.array(ratio)
            ok = np.abs(err) < FAIL
            n = int(ok.sum())
            lo, hi = np.percentile(ratio[ok], [25, 75]) if n else (np.nan, np.nan)
            row = dict(circuits=m, schedule=kind, trials=TRIALS, fail_rate=round(1 - ok.mean(), 4),
                       rmse_khz=round(float(np.sqrt((err[ok] ** 2).mean())) / 1e3, 3) if n else "",
                       crlb_khz=round(float(np.median(bound)) / 1e3, 3),
                       t2_ratio_q1=round(float(lo), 2), t2_ratio_q3=round(float(hi), 2))
            rows.append(row)
            print(row, f"{time.time() - t0:.0f}s", flush=True)
    with open(ROOT / "results" / "benchmark.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
