import math
import random

import pytest

from membrane_crystallization.linalg import eigh_hermitian, fidelity, zeros
from membrane_crystallization.stream import (crystallize, hebbian_update,
                                             match_fidelity, noisy,
                                             pattern_sampler, residue_whiteness,
                                             self_recreate, spectrum_entropy,
                                             stream, subspace_fidelity)
from membrane_crystallization.sweeps import FREQS, PATTERNS

S2 = 2 ** -0.5


def test_ema_is_exponentially_weighted_covariance():
    rng = random.Random(1)
    xs = [noisy([1, 0, 0], 0.5, rng) for _ in range(30)]
    W = zeros(3)
    for x in xs:
        hebbian_update(W, x, 0.1)
    T = len(xs)
    for a in range(3):
        for b in range(3):
            want = sum(0.1 * 0.9 ** (T - 1 - t) * xs[t][a] * xs[t][b].conjugate() for t in range(T))
            assert abs(W[a][b] - want) < 1e-12


def test_trace_tracks_one_minus_decay():
    rng = random.Random(2)
    W, _ = stream(lambda t: noisy([1, 0], 0.3, rng), 50, 0.1)
    assert abs((W[0][0] + W[1][1]).real - (1 - 0.9 ** 50)) < 1e-12


def test_entropy_extremes():
    assert spectrum_entropy([1, 1, 1, 1]) == pytest.approx(math.log(4))
    assert spectrum_entropy([1, 0, 0, 0]) == pytest.approx(0)
    with pytest.raises(ValueError):
        spectrum_entropy([0, 0])


def test_single_pattern_learned():
    p = [S2, 0j, 0j, S2]
    W, _ = crystallize([p], 1, 2000, 0.25, 0.02, seed=5)
    _, vecs = eigh_hermitian(W)
    assert fidelity(vecs[0], p) > 0.99


def test_three_patterns_recovered():
    W, snaps = crystallize(PATTERNS, 3, 3000, 0.25, 0.01, seed=7, freqs=FREQS,
                           snapshot_at=[1, 50, 3000])
    assert [s[0] for s in snaps] == [1, 50, 3000]
    assert snaps[0][1] == pytest.approx(0, abs=1e-9)    # rank one after one step
    assert snaps[-1][2] > 0.93 and snaps[-1][3] > 0.9
    assert residue_whiteness(W, 3) > 0.97


def test_metrics():
    e = [[1, 0, 0], [0, 1, 0]]
    t = [[S2, S2, 0], [0, 0, 1]]
    assert subspace_fidelity(e, t) == pytest.approx(0.5)
    assert match_fidelity(e, [[1, 0, 0]]) == pytest.approx(1)
    with pytest.raises(ValueError):
        match_fidelity([[1, 0]], [[1, 0], [0, 1]])


def test_deterministic():
    a = crystallize(PATTERNS, 3, 300, 0.3, 0.05, seed=9, freqs=FREQS, snapshot_at=[150, 300])
    b = crystallize(PATTERNS, 3, 300, 0.3, 0.05, seed=9, freqs=FREQS, snapshot_at=[150, 300])
    assert a[1] == b[1]


def test_self_recreation_shape():
    out = self_recreate(PATTERNS, 3, 3, 1500, 0.25, 0.01, freqs=FREQS, seed=3)
    assert len(out) == 3 and all(0 <= m <= 1 and 0 <= s <= 1 for m, s in out)


def test_k_too_small_gives_no_match_fidelity():
    _, snaps = crystallize(PATTERNS, 2, 100, 0.25, 0.05, seed=1, freqs=FREQS, snapshot_at=[100])
    assert snaps[0][3] is None


@pytest.mark.parametrize("kwargs", [
    dict(patterns=[]),
    dict(patterns=[[1, 0], [1, 0, 0]]),
    dict(patterns=[[0, 0]]),
    dict(k=0), dict(k=9), dict(k=1.5),
    dict(freqs=[0.5, 0.5]), dict(freqs=[0.5, 0.3, 0.1]), dict(freqs=[1.2, -0.1, -0.1]),
    dict(alpha=0), dict(alpha=1.5), dict(steps=0), dict(sigma=-1),
])
def test_crystallize_validation(kwargs):
    args = dict(patterns=PATTERNS, k=3, steps=10, sigma=0.2, alpha=0.1, seed=0, freqs=FREQS)
    args.update(kwargs)
    with pytest.raises(ValueError):
        crystallize(**args)


def test_other_validation():
    W = zeros(4)
    with pytest.raises(ValueError):
        hebbian_update(W, [1, 0], 0.1)
    W[0][0] = 1
    with pytest.raises(ValueError):
        residue_whiteness(W, 3)          # needs N - k >= 2
    with pytest.raises(ValueError):
        residue_whiteness(W, 2)          # zero residue trace
    with pytest.raises(ValueError):
        self_recreate(PATTERNS, 2, 1, 10, 0.2, 0.1, freqs=FREQS)
    with pytest.raises(ValueError):
        noisy([1, 0], 0.1, random.Random(0), scales=[1])
    with pytest.raises(ValueError):
        pattern_sampler(PATTERNS, 0.1, random.Random(0), freqs=[1, 0])
