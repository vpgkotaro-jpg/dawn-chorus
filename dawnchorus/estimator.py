"""Grid least-squares fit of a Ramsey signal, plus the Cramer-Rao bound on detuning."""
import numpy as np

DETUNING_GRID = np.arange(-450e3, 450e3, 0.5e3)
T2_GRID = np.geomspace(15e-6, 500e-6, 10)


def signal(taus, delta, t2, f0, amp=-0.5, offset=0.5):
    return offset + amp * np.exp(-taus / t2) * np.cos(2 * np.pi * (f0 + delta) * taus)


def fit(taus, p1, f0):
    """Estimate (detuning, T2) from excited-state fractions.

    Amplitude and offset are profiled out analytically, so SPAM needs no separate calibration.
    """
    y = p1 - p1.mean()
    best = (np.inf, 0.0, 0.0)
    for t2 in T2_GRID:
        c = np.cos(2 * np.pi * np.outer(f0 + DETUNING_GRID, taus)) * np.exp(-taus / t2)
        c = c - c.mean(axis=1, keepdims=True)
        sse = (y * y).sum() - (c * y).sum(axis=1) ** 2 / ((c * c).sum(axis=1) + 1e-12)
        i = sse.argmin()
        if sse[i] < best[0]:
            best = (sse[i], DETUNING_GRID[i], t2)
    return _refine(taus, y, f0, best[1], best[2])


def candidates(taus, p1, f0, tol=0.02, sep=10e3):
    """Return detunings that explain the data about as well as the best one, as (delta, R^2).

    On a uniform schedule the aliases are exact ties, so choosing among them is arbitrary.
    """
    y = p1 - p1.mean()
    total = (y * y).sum()
    r2 = np.zeros_like(DETUNING_GRID)
    for t2 in T2_GRID:
        c = np.cos(2 * np.pi * np.outer(f0 + DETUNING_GRID, taus)) * np.exp(-taus / t2)
        c = c - c.mean(axis=1, keepdims=True)
        r2 = np.maximum(r2, (c * y).sum(axis=1) ** 2 / ((c * c).sum(axis=1) + 1e-12) / total)
    out = []
    for i in np.argsort(-r2):
        if r2[i] < r2.max() - tol:
            break
        if all(abs(DETUNING_GRID[i] - d) > sep for d, _ in out):
            out.append((float(DETUNING_GRID[i]), float(r2[i])))
    return sorted(out)


def _refine(taus, y, f0, delta, t2):
    """Polish the coarse optimum on a fine grid. y must already be mean-subtracted."""
    deltas = delta + np.arange(-1e3, 1e3 + 1, 25.0)
    t2s = t2 * np.geomspace(0.65, 1.55, 31)
    phase = np.cos(2 * np.pi * np.outer(f0 + deltas, taus))
    best = (np.inf, delta, t2)
    for t in t2s:
        c = phase * np.exp(-taus / t)
        c = c - c.mean(axis=1, keepdims=True)
        sse = (y * y).sum() - (c * y).sum(axis=1) ** 2 / ((c * c).sum(axis=1) + 1e-12)
        i = sse.argmin()
        if sse[i] < best[0]:
            best = (sse[i], deltas[i], t)
    return best[1], best[2]


def crlb(taus, t2, shots, f0, deltas=np.linspace(-300e3, 300e3, 41), amp=0.5):
    """Detuning standard deviation bound in Hz, assuming T2, amplitude and offset are known.

    Averaged over the true detuning. Real fits estimate those too, so this is a floor.
    """
    out = []
    for d in deltas:
        phase = 2 * np.pi * (f0 + d) * taus
        env = amp * np.exp(-taus / t2)
        p = np.clip(0.5 - env * np.cos(phase), 1e-6, 1 - 1e-6)
        dp = env * np.sin(phase) * 2 * np.pi * taus
        out.append(shots * np.sum(dp ** 2 / (p * (1 - p))))
    return float(np.sqrt(1.0 / np.mean(out)))
