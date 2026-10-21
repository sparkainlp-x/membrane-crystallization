"""Small standard-library linear algebra for complex Hermitian matrices.

eigh_hermitian embeds H = A + iB in the real symmetric matrix
[[A, -B], [B, A]] and diagonalizes it with cyclic Jacobi. Every
eigenvalue of H appears twice in the embedding. Within each cluster of
(numerically) equal eigenvalues, the real eigenvectors are mapped back to
complex vectors and Gram-Schmidt keeps an orthonormal set of the right
size. This handles repeated eigenvalues of H, where simply taking every
other real eigenvector can return duplicates (the draft did; see
spec/crystal.py and tests/test_linalg.py).
"""

import cmath
import math


def _check_vector(v, name="vector"):
    v = [complex(c) for c in v]
    if not v:
        raise ValueError(f"{name} is empty")
    if not all(cmath.isfinite(c) for c in v):
        raise ValueError(f"{name} has non-finite entries")
    return v


def norm(v):
    return math.sqrt(sum(abs(c) ** 2 for c in v))


def normalize(v):
    v = _check_vector(v)
    n = norm(v)
    if n == 0.0:
        raise ValueError("cannot normalize a zero-norm vector")
    return [c / n for c in v]


def inner(a, b):
    """<a|b>."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(x.conjugate() * y for x, y in zip(a, b))


def fidelity(a, b):
    return abs(inner(a, b)) ** 2


def zeros(n):
    return [[0j] * n for _ in range(n)]


def check_hermitian(H, tol=1e-9):
    n = len(H)
    if n == 0 or any(len(row) != n for row in H):
        raise ValueError("matrix must be square and non-empty")
    for i in range(n):
        for j in range(n):
            if abs(H[i][j] - complex(H[j][i]).conjugate()) > tol:
                raise ValueError("matrix is not Hermitian")


def jacobi_real(A, tol=1e-14, max_sweeps=100):
    """Eigenvalues and eigenvector columns of a real symmetric matrix."""
    n = len(A)
    A = [list(map(float, row)) for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = max((abs(x) for row in A for x in row), default=0.0) or 1.0
    for _ in range(max_sweeps):
        off = math.sqrt(sum(A[p][q] ** 2 for p in range(n) for q in range(n) if p != q))
        if off <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) <= 1e-300:
                    continue
                tau = (A[q][q] - A[p][p]) / (2.0 * apq)
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + math.sqrt(1 + tau * tau))
                c = 1.0 / math.sqrt(1 + t * t)
                s = t * c
                for k in range(n):
                    akp, akq = A[k][p], A[k][q]
                    A[k][p] = c * akp - s * akq
                    A[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = A[p][k], A[q][k]
                    A[p][k] = c * apk - s * aqk
                    A[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = V[k][p], V[k][q]
                    V[k][p] = c * vkp - s * vkq
                    V[k][q] = s * vkp + c * vkq
    else:
        raise RuntimeError("Jacobi did not converge")
    return [A[i][i] for i in range(n)], V


def eigh_hermitian(H, cluster_tol=1e-8):
    """Eigenvalues (descending) and orthonormal eigenvectors of Hermitian H."""
    check_hermitian(H)
    n = len(H)
    M = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            c = complex(H[i][j])
            M[i][j] = c.real
            M[i][j + n] = -c.imag
            M[i + n][j] = c.imag
            M[i + n][j + n] = c.real
    vals, V = jacobi_real(M)
    order = sorted(range(2 * n), key=lambda i: vals[i], reverse=True)
    scale = max(1.0, max(abs(v) for v in vals))
    clusters, cur = [], [order[0]]
    for i in order[1:]:
        if abs(vals[cur[-1]] - vals[i]) <= cluster_tol * scale:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)
    eigvals, eigvecs = [], []
    for cl in clusters:
        if len(cl) % 2:
            raise RuntimeError("eigenvalue twins were split; loosen cluster_tol")
        kept = []
        for i in cl:
            v = [complex(V[k][i], V[k + n][i]) for k in range(n)]
            for u in kept:
                m = inner(u, v)
                v = [a - m * b for a, b in zip(v, u)]
            nv = norm(v)
            if nv > 1e-6:
                kept.append([a / nv for a in v])
            if len(kept) == len(cl) // 2:
                break
        if len(kept) != len(cl) // 2:
            raise RuntimeError("could not recover a full eigenbasis")
        for u in kept:
            Hu = [sum(H[a][b] * u[b] for b in range(n)) for a in range(n)]
            eigvals.append(inner(u, Hu).real)
            eigvecs.append(u)
    return eigvals, eigvecs


def reconstruct(vals, vecs):
    n = len(vecs[0])
    return [[sum(l * v[a] * v[b].conjugate() for l, v in zip(vals, vecs))
             for b in range(n)] for a in range(n)]
