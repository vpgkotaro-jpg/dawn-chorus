import numpy as np

from dawnchorus import estimator, schedules as sc

F0 = 400e3


def test_uniform_aliases_exactly():
    taus = sc.uniform(16, 40e-6)
    assert sc.ambiguity(taus)[1] > 0.999


def test_chirp_breaks_aliasing():
    assert sc.ambiguity(sc.chirp(16, 40e-6))[1] < 0.6


def test_fit_recovers_noiseless_signal():
    taus = sc.chirp(16, 40e-6)
    d, t2 = estimator.fit(taus, estimator.signal(taus, 123e3, 80e-6, F0), F0)
    assert abs(d - 123e3) <= 1e3
    assert 0.7 < t2 / 80e-6 < 1.4


def test_uniform_fit_is_fooled_by_an_alias():
    taus = sc.uniform(16, 40e-6)
    alias = 1 / (taus[1] - taus[0])
    a = estimator.signal(taus, -100e3, 80e-6, F0)
    b = estimator.signal(taus, -100e3 + alias, 80e-6, F0)
    assert np.allclose(a, b)


def test_crlb_shrinks_with_shots():
    taus = sc.chirp(16, 40e-6)
    assert estimator.crlb(taus, 80e-6, 1024, F0) < estimator.crlb(taus, 80e-6, 256, F0)


def test_uniform_scan_has_several_tied_answers():
    taus = sc.uniform(16, 40e-6)
    answers = estimator.candidates(taus, estimator.signal(taus, 180e3, 80e-6, F0), F0)
    assert any(abs(d - 180e3) < 1e3 for d, _ in answers)
    assert len(answers) == 4


def test_chirp_scan_has_one_answer():
    taus = sc.chirp(16, 40e-6)
    answers = estimator.candidates(taus, estimator.signal(taus, 180e3, 80e-6, F0), F0)
    assert len(answers) == 1
