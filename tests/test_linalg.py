import cmath
import math
import random

import pytest

from membrane_crystallization.linalg import (eigh_hermitian, inner, jacobi_real,
                                             normalize, reconstruct, zeros)

S2 = 2 ** -0.5


def _check(H, vals=None, tol=1e-9):
    n = len(H)
    ev, V = eigh_hermitian(H)
    assert len(ev) == n and len(V) == n
    for i in range(n):
        for j in range(n):
            assert abs(inner(V[i], V[j]) - (1 if i == j else 0)) < tol
    R = reconstruct(ev, V)
    assert max(abs(R[a][b] - H[a][b]) for a in range(n) for b in range(n)) < tol
    assert ev == sorted(ev, reverse=True)
    if vals is not None:
        assert ev == pytest.approx(sorted(vals, reverse=True), abs=tol)
    return ev, V


def _unitary(n, seed):
    rng = random.Random(seed)
    U = []
    for _ in range(n):
        v = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(n)]
        for u in U:
            m = inner(u, v)
            v = [a - m * b for a, b in zip(v, u)]
        U.append(normalize(v))
    return U


def _from_spectrum(vals, U):
    return reconstruct(vals, U)


def test_jacobi_known():
    vals, _ = jacobi_real([[2.0, 1.0], [1.0, 2.0]])
    assert sorted(vals) == pytest.approx([1.0, 3.0])


def test_identity_fully_degenerate():
    _check([[1 + 0j if i == j else 0j for j in range(4)] for i in range(4)], [1] * 4)


def test_uniform_bell_projector():
    P = [[S2, 0, 0, S2], [S2, 0, 0, -S2], [0, S2, S2, 0]]
    H = zeros(4)
    for p in P:
        for a in range(4):
            for b in range(4):
                H[a][b] += p[a] * p[b] / 3
    ev, V = _check(H, [1 / 3] * 3 + [0])
    span = sum(sum(abs(inner(v, p)) ** 2 for v in V[:3]) for p in P) / 3
    assert span == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("vals", [[3, 3, 1, 1, 1, 0], [2, 2, 2, 2, -1, -1],
                                  [5, 4, 3, 2, 1, 0], [0.5] * 6])
def test_known_spectra_in_random_bases(seed, vals):
    U = _unitary(6, seed)
    H = _from_spectrum(vals, U)
    ev, V = _check(H, vals)
    # each eigenspace is spanned correctly
    for lam in set(vals):
        true = [u for u, l in zip(U, vals) if l == lam]
        got = [v for v, l in zip(V, ev) if abs(l - lam) < 1e-7]
        assert len(got) == len(true)
        for t in true:
            assert sum(abs(inner(g, t)) ** 2 for g in got) == pytest.approx(1, abs=1e-9)


def test_complex_phase_matrix():
    H = [[2, 1j], [-1j, 2]]
    ev, V = _check(H, [3, 1])
    assert abs(abs(inner(V[0], [S2, -1j * S2])) - 1) < 1e-9


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        eigh_hermitian([[1, 1j], [1j, 1]])
    with pytest.raises(ValueError):
        eigh_hermitian([[1, 0]])
    with pytest.raises(ValueError):
        eigh_hermitian([])
    with pytest.raises(ValueError):
        normalize([0, 0])
    with pytest.raises(ValueError):
        normalize([])
    with pytest.raises(ValueError):
        normalize([cmath.nan])
    with pytest.raises(ValueError):
        inner([1], [1, 0])


def test_draft_solver_fails_degenerate_case():
    """Documents the bug fixed here: the draft's twin pairing returned
    duplicate vectors for the identity matrix."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parents[1] / "spec" / "crystal.py"
    spec = importlib.util.spec_from_file_location("draft_crystal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    I = [[1 + 0j if i == j else 0j for j in range(4)] for i in range(4)]
    _, V = mod.eigh_hermitian(I)
    gram = max(abs(abs(inner(V[i], V[j])) - (i == j)) for i in range(4) for j in range(4))
    assert gram > 0.5
    assert math.isfinite(gram)
