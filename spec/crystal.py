"""Crystallization: a holographic membrane that learns patterns from a
noisy stream, watches its own entropy, and re-derives itself.

No clean patterns are handed to the membrane. It sees a stream of
noisy observations and integrates them Hebbian-style:

    W_t = (1 - alpha) * W_{t-1} + alpha * |x_t><x_t|

Crystallization is measured three ways:
  1. Spectrum entropy S(t) of W_t (von Neumann, normalized).
  2. Extraction fidelity: top-K eigenvectors vs the true patterns,
     sampled over time -- the crystallization curve.
  3. Residue whiteness: after removing the top-K components, the
     residue's entropy per unit trace should be ~maximal (white):
     the membrane extracted all the structure, leaving noise.

Self-recreation: generation g+1 crystallizes from noisy observations
of generation g's *extracted* patterns (not the originals). Fidelity
to the true patterns is measured per generation -- the actual horizon
of "infinite" self-recreation, not the slogan.

Eigenvalues of complex Hermitian matrices go through the real
symmetric embedding [[Re, -Im], [Im, Re]] with a cyclic Jacobi
solver (stdlib only).
"""

import math
import random


# ---------------------------------------------------------------- utils
def normalize(v):
    n = math.sqrt(sum(abs(c) ** 2 for c in v))
    return [c / n for c in v]


def fidelity(a, b):
    return abs(sum(x.conjugate() * y for x, y in zip(a, b))) ** 2


def noisy(state, sigma, rng):
    return normalize([c + complex(rng.gauss(0, sigma), rng.gauss(0, sigma))
                      for c in state])


def zeros(n):
    return [[0j] * n for _ in range(n)]


def hebbian_update(W, x, alpha):
    """W <- (1-alpha) W + alpha |x><x|  (in place)."""
    n = len(W)
    for i in range(n):
        xi = x[i]
        Wi = W[i]
        for j in range(n):
            Wi[j] = (1 - alpha) * Wi[j] + alpha * xi * x[j].conjugate()
    return W


# ------------------------------------------------- Jacobi eigensolver
def _jacobi_real(A, tol=1e-13, max_sweeps=200):
    """Eigenvalues/vectors of a real symmetric matrix (cyclic Jacobi)."""
    n = len(A)
    A = [row[:] for row in A]
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        changed = False
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = A[p][q]
                if abs(apq) > tol:
                    changed = True
                    app, aqq = A[p][p], A[q][q]
                    tau = (aqq - app) / (2.0 * apq)
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
        if not changed:
            break
    return [A[i][i] for i in range(n)], V


def eigh_hermitian(H):
    """Eigenvalues (desc) and eigenvectors of complex Hermitian H."""
    n = len(H)
    M = [[0.0] * (2 * n) for _ in range(2 * n)]
    for i in range(n):
        for j in range(n):
            c = H[i][j]
            M[i][j] = c.real
            M[i][j + n] = -c.imag
            M[i + n][j] = c.imag
            M[i + n][j + n] = c.real
    vals, vecs = _jacobi_real(M)
    # each eigenvalue appears twice; take every other after sorting desc
    order = sorted(range(2 * n), key=lambda i: vals[i], reverse=True)
    eigvals, eigvecs = [], []
    used = [False] * (2 * n)
    for i in order:
        if used[i]:
            continue
        used[i] = True
        # mark its pair twin (nearest unused with ~equal eigenvalue)
        for j in order:
            if not used[j] and abs(vals[j] - vals[i]) < 1e-9:
                used[j] = True
                break
        eigvals.append(vals[i])
        eigvecs.append([complex(vecs[k][i], vecs[k + n][i])
                        for k in range(n)])
    eigvecs = [normalize(v) for v in eigvecs]
    return eigvals, eigvecs


def spectrum_entropy(eigvals):
    """Von Neumann entropy of the normalized spectrum (nats)."""
    tot = sum(eigvals)
    s = 0.0
    for l in eigvals:
        p = l / tot
        if p > 1e-15:
            s -= p * math.log(p)
    return s


# ------------------------------------------------- crystallization
def crystallize(patterns, k, steps, sigma, alpha, seed,
                freqs=None, snapshot_every=None):
    """Stream noisy observations of `patterns`; return final W plus
    (t, entropy, subspace_fidelity, match_fidelity) snapshots.

    freqs: observation probabilities per pattern (default uniform).
    Distinct frequencies separate the eigenvalues, making individual
    patterns identifiable -- with uniform frequencies the top-K
    eigenspace is right but its basis is arbitrary (degeneracy).
    """
    n = len(patterns[0])
    rng = random.Random(seed)
    if freqs is None:
        freqs = [1.0 / len(patterns)] * len(patterns)
    cum = []
    s = 0.0
    for f in freqs:
        s += f
        cum.append(s)
    W = zeros(n)
    snaps = []
    for t in range(1, steps + 1):
        r = rng.random()
        p = patterns[next(i for i, c in enumerate(cum) if r < c)]
        hebbian_update(W, noisy(p, sigma, rng), alpha)
        if snapshot_every and t % snapshot_every == 0:
            vals, vecs = eigh_hermitian(W)
            snaps.append((t, spectrum_entropy(vals),
                          subspace_fidelity(vecs[:k], patterns),
                          match_fidelity(vecs[:k], patterns)))
    return W, snaps


def subspace_fidelity(extracted, true):
    """Mean squared projection of each true pattern onto the span of
    the extracted top-K: the principled metric for what Hebbian
    learning actually recovers (the subspace, not a basis)."""
    tot = 0.0
    for t in true:
        tot += sum(abs(sum(v[i].conjugate() * t[i]
                           for i in range(len(t)))) ** 2
                   for v in extracted)
    return tot / len(true)


def match_fidelity(extracted, true):
    """Greedy best-matching mean fidelity (order-free)."""
    remaining = list(range(len(extracted)))
    tot = 0.0
    for t in true:
        j = max(remaining, key=lambda j: fidelity(extracted[j], t))
        remaining.remove(j)
        tot += fidelity(extracted[j], t)
    return tot / len(true)


def residue_whiteness(W, k):
    """Entropy of the residue (W minus top-k) per unit trace, as a
    fraction of the maximum log(N-k). ~1.0 means the residue is white:
    all structure was extracted."""
    n = len(W)
    vals, vecs = eigh_hermitian(W)
    R = [row[:] for row in W]
    for i in range(k):
        l, v = vals[i], vecs[i]
        for a in range(n):
            for b in range(n):
                R[a][b] -= l * v[a] * v[b].conjugate()
    rvals, _ = eigh_hermitian(R)
    rvals = [max(0.0, l) for l in rvals]  # clip fp noise; R is PSD
    if sum(rvals) == 0:
        return 1.0
    return spectrum_entropy(rvals) / math.log(n - k)


def self_recreate(patterns, k, generations, steps, sigma, alpha,
                  freqs=None, seed=0):
    """Generation g+1 crystallizes from noisy observations of generation
    g's extracted patterns. Returns fidelity-to-true per generation."""
    cur = [normalize(list(p)) for p in patterns]
    fids = []
    for g in range(generations):
        W, _ = crystallize(cur, k, steps, sigma, alpha,
                           freqs=freqs, seed=seed + 1000 * g)
        vals, vecs = eigh_hermitian(W)
        fids.append(match_fidelity(vecs[:k], patterns))
        cur = vecs[:k]
    return fids
