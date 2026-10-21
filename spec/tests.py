"""Tests for membrane-crystallization. Run: python tests.py"""

import math
import random

from crystal import (normalize, fidelity, noisy, zeros, hebbian_update,
                     _jacobi_real, eigh_hermitian, spectrum_entropy,
                     crystallize, match_fidelity, subspace_fidelity,
                     residue_whiteness, self_recreate)

S2 = 2 ** -0.5


def test_jacobi_known():
    vals, _ = _jacobi_real([[2.0, 1.0], [1.0, 2.0]])
    assert abs(sorted(vals)[0] - 1.0) < 1e-9
    assert abs(sorted(vals)[1] - 3.0) < 1e-9


def test_hermitian_reconstruction():
    rng = random.Random(3)
    n = 4
    H = zeros(n)
    for _ in range(50):
        x = normalize([complex(rng.gauss(0, 1), rng.gauss(0, 1))
                       for _ in range(n)])
        hebbian_update(H, x, 0.1)
    vals, vecs = eigh_hermitian(H)
    assert len(vals) == n and len(vecs) == n
    # H ~= V L V^dagger
    err = 0.0
    for a in range(n):
        for b in range(n):
            rec = sum(vals[i] * vecs[i][a] * vecs[i][b].conjugate()
                      for i in range(n))
            err = max(err, abs(rec - H[a][b]))
    assert err < 1e-9, err
    # eigenvectors orthonormal
    for i in range(n):
        for j in range(n):
            ip = abs(sum(vecs[i][k].conjugate() * vecs[j][k]
                         for k in range(n)))
            assert abs(ip - (1.0 if i == j else 0.0)) < 1e-9


def test_entropy_extremes():
    assert abs(spectrum_entropy([1.0, 1.0, 1.0, 1.0]) - math.log(4)) < 1e-12
    assert abs(spectrum_entropy([1.0, 0.0, 0.0, 0.0])) < 1e-12


def test_single_pattern_crystallizes():
    p = [S2, 0j, 0j, S2]
    W, _ = crystallize([p], 1, 2000, 0.25, 0.02, seed=5)
    vals, vecs = eigh_hermitian(W)
    assert fidelity(vecs[0], p) > 0.99


def test_three_patterns_recovered():
    pats = [[S2, 0j, 0j, S2] + [0j] * 4,
            [S2, 0j, 0j, -S2] + [0j] * 4,
            [0j, S2, S2, 0j] + [0j] * 4]
    W, _ = crystallize(pats, 3, 6000, 0.25, 0.01, seed=7,
                       freqs=[0.6, 0.3, 0.1])
    vals, vecs = eigh_hermitian(W)
    assert subspace_fidelity(vecs[:3], pats) > 0.95
    assert match_fidelity(vecs[:3], pats) > 0.85
    assert residue_whiteness(W, 3) > 0.90


def test_deterministic():
    pats = [[S2, 0j, 0j, S2], [S2, 0j, 0j, -S2]]
    _, s1 = crystallize(pats, 2, 500, 0.3, 0.05, seed=9, snapshot_every=250)
    _, s2 = crystallize(pats, 2, 500, 0.3, 0.05, seed=9, snapshot_every=250)
    assert s1 == s2


def test_self_recreation_holds_two_generations():
    pats = [[S2, 0j, 0j, S2] + [0j] * 4,
            [S2, 0j, 0j, -S2] + [0j] * 4,
            [0j, S2, S2, 0j] + [0j] * 4]
    fids = self_recreate(pats, 3, 2, 6000, 0.25, 0.01,
                         freqs=[0.6, 0.3, 0.1], seed=7)
    assert fids[0] > 0.85 and fids[1] > 0.80


if __name__ == "__main__":
    for name, fn in sorted(
            [(k, v) for k, v in globals().items() if k.startswith("test_")]):
        fn()
        print(f"PASS {name}")
    print("all tests passed")
