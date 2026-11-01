"""Delay schedules for a Ramsey scan, and the ambiguity function that compares them."""
import numpy as np

TAU_MIN = 0.25e-6


def uniform(m, tau_max):
    """Evenly spaced delays. Aliases every 1 / spacing."""
    return tau_max * np.arange(1, m + 1) / m


def uniform_nyquist(m, f_max):
    """Evenly spaced delays, short enough that f_max does not alias. Fair uniform baseline."""
    return uniform(m, m / (2.0 * f_max))


def two_stage(m, tau_max, f_max):
    """Half the circuits short and alias-free, half long and precise. The usual practical recipe."""
    short = m // 2
    return np.sort(np.concatenate([uniform_nyquist(short, f_max), uniform(m - short, tau_max)]))


def chirp(m, tau_max):
    """Quadratic delays: the sampling rate sweeps from high to low, like a falling chirp."""
    return tau_max * (np.arange(1, m + 1) / m) ** 2


def geometric(m, tau_max, tau_min=TAU_MIN):
    """Log-spaced delays, the spacing used by robust phase estimation."""
    return np.geomspace(tau_min, tau_max, m)


def random_uniform(m, tau_max, rng, tau_min=TAU_MIN):
    """Fresh random delays for each scan, the compressed-sensing baseline."""
    return np.sort(rng.uniform(tau_min, tau_max, m))


def ambiguity(taus, span=800e3, points=6401):
    """Return (half-width in Hz, peak sidelobe) of |mean exp(2j*pi*d*tau)| over detuning d.

    The half-width sets precision. A sidelobe of 1 means two detunings are indistinguishable.
    """
    d = np.linspace(-span, span, points)
    a = np.abs(np.exp(2j * np.pi * np.outer(d, taus)).mean(axis=1))
    half = np.argmax(a[points // 2:] < 0.5) * (d[1] - d[0])
    return float(half), float(a[np.abs(d) > 2 * half].max())
