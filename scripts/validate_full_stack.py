"""End-to-end check: native gates, ALAP scheduling, the backend's own noise model, SamplerV2."""
import csv
import pathlib
import sys
import warnings

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dawnchorus import device, estimator, schedules as sc

warnings.filterwarnings("ignore")
TAU_MAX, F0, SHOTS, M = 40e-6, 400e3, 256, 16
QUBITS, REPEATS = 5, 4
BACKENDS = ("fake_cairo", "fake_hanoi", "fake_sherbrooke", "fake_kyiv", "fake_torino", "fake_marrakesh")


def main():
    rng = np.random.default_rng(7)
    provider = FakeProviderForBackendV2()
    rows = []
    for name in BACKENDS:
        b = provider.backend(name)
        noise = device.full_stack_noise(b)
        good = [q for q in device.sample_qubits(rng, per_backend=b.num_qubits) if q.backend == name]
        for q in [good[i] for i in rng.permutation(len(good))[:QUBITS]]:
            for _ in range(REPEATS):
                delta = rng.uniform(-300e3, 300e3)
                taus = sc.chirp(M, TAU_MAX)
                sampler = device.full_stack_sampler(noise, seed=int(rng.integers(2 ** 31)))
                y, circuits = device.full_stack_scan(b, sampler, q.index, taus, delta, F0, SHOTS)
                d_hat, t2_hat = estimator.fit(taus, y, F0)
                rows.append(dict(backend=name, qubit=q.index, true_delta_khz=round(delta / 1e3, 2),
                                 err_khz=round(float(d_hat - delta) / 1e3, 3),
                                 t2_reported_us=round(q.t2 * 1e6, 1), t2_fit_us=round(float(t2_hat) * 1e6, 1)))
        print(name, len(rows), flush=True)
    with open(ROOT / "results" / "full_stack.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    e = np.array([r["err_khz"] for r in rows])
    fit, reported = np.array([r["t2_fit_us"] for r in rows]), np.array([r["t2_reported_us"] for r in rows])
    print("n", len(e), "fail", int((np.abs(e) > 10).sum()),
          "rmse_ok_khz", round(float(np.sqrt((e[np.abs(e) <= 10] ** 2).mean())), 3))
    for label, keep in (("all", reported > 0), ("T2<=80us", reported <= 80), ("T2>80us", reported > 80)):
        print("t2_ratio", label, int(keep.sum()), *np.round(np.percentile(fit[keep] / reported[keep], [25, 50, 75]), 2))
    # How risky is one draw of the random schedule? Sidelobe for the fixed chirp vs many draws.
    for m in (8, 16):
        s = [sc.ambiguity(sc.random_uniform(m, TAU_MAX, rng))[1] for _ in range(500)]
        print(m, "chirp sidelobe", round(sc.ambiguity(sc.chirp(m, TAU_MAX))[1], 3),
              "random median/p95/max", *np.round(np.percentile(s, [50, 95, 100]), 3))


if __name__ == "__main__":
    main()
