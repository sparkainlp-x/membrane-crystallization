"""Streaming Hebbian estimate of an observation covariance.

SYNTHETIC. W_t = (1 - alpha) W_{t-1} + alpha |x_t><x_t| is an
exponentially weighted running estimate of the second-moment
(covariance) matrix E[|x><x|] of the observations. Its top-K
eigenvectors are the estimate's principal components: this is
streaming PCA, the setting of Oja (1982). "Crystallization" is used in
this repo only as a descriptive label for that estimate settling.

With a fixed alpha the estimate forgets: an observation seen s steps
ago carries weight alpha (1 - alpha)^s, so the effective memory window
is about 1/alpha steps (weight falls by 1/e after 1/alpha steps).
"""

import math
import random

from .linalg import eigh_hermitian, fidelity, inner, normalize, zeros


def check_patterns(patterns):
    patterns = list(patterns)
    if not patterns:
        raise ValueError("need at least one pattern")
    out = [normalize(p) for p in patterns]
    n = len(out[0])
    if any(len(p) != n for p in out):
        raise ValueError("all patterns must have the same length")
    return out


def check_freqs(freqs, m):
    if freqs is None:
        return [1.0 / m] * m
    freqs = [float(f) for f in freqs]
    if len(freqs) != m:
        raise ValueError(f"need {m} frequencies, got {len(freqs)}")
    if any(not math.isfinite(f) or f < 0 for f in freqs):
        raise ValueError("frequencies must be finite and non-negative")
    if abs(sum(freqs) - 1.0) > 1e-9:
        raise ValueError(f"frequencies must sum to 1, got {sum(freqs)}")
    return freqs


def check_alpha(alpha):
    if not (isinstance(alpha, (int, float)) and 0 < alpha <= 1):
        raise ValueError("alpha must be in (0, 1]")


def check_k(k, n, upper=None):
    upper = n if upper is None else upper
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= upper:
        raise ValueError(f"k must be an integer in [1, {upper}], got {k!r}")


def noisy(state, sigma, rng, scales=None):
    """state + complex Gaussian noise, renormalized.

    scales: optional per-component noise multipliers (anisotropic noise).
    """
    if sigma < 0:
        raise ValueError("sigma must be >= 0")
    if scales is None:
        scales = [1.0] * len(state)
    elif len(scales) != len(state):
        raise ValueError("scales length mismatch")
    return normalize([c + s * complex(rng.gauss(0, sigma), rng.gauss(0, sigma))
                      for c, s in zip(state, scales)])


def hebbian_update(W, x, alpha):
    """W <- (1 - alpha) W + alpha |x><x| (in place)."""
    if len(x) != len(W):
        raise ValueError("observation length mismatch")
    b = 1.0 - alpha
    xc = [c.conjugate() for c in x]
    for i, xi in enumerate(x):
        a = alpha * xi
        W[i] = [b * w + a * c for w, c in zip(W[i], xc)]
    return W


def spectrum_entropy(eigvals):
    """Von Neumann entropy (nats) of the trace-normalized spectrum."""
    vals = [max(0.0, v) for v in eigvals]
    tot = sum(vals)
    if tot <= 0:
        raise ValueError("spectrum has zero trace")
    return -sum((v / tot) * math.log(v / tot) for v in vals if v / tot > 1e-15)


def subspace_fidelity(extracted, true):
    """Mean squared norm of each true pattern's projection onto span(extracted).

    `extracted` must be orthonormal (eigenvectors are).
    """
    return sum(sum(abs(inner(v, t)) ** 2 for v in extracted) for t in true) / len(true)


def match_fidelity(extracted, true):
    """Greedy one-to-one best match, mean fidelity."""
    if len(extracted) < len(true):
        raise ValueError("need at least as many extracted vectors as true patterns")
    remaining = list(range(len(extracted)))
    tot = 0.0
    for t in true:
        j = max(remaining, key=lambda j: fidelity(extracted[j], t))
        remaining.remove(j)
        tot += fidelity(extracted[j], t)
    return tot / len(true)


def residue_whiteness(W, k):
    """Entropy of the spectrum left after removing the top k, over log(N - k).

    Near 1 means the remaining spectrum is flat. With isotropic noise that
    is true by construction, including for a stream with no patterns at
    all; see the controls in sweeps.whiteness_controls.
    """
    n = len(W)
    check_k(k, n, upper=n - 2)
    vals, _ = eigh_hermitian(W)
    rest = [max(0.0, v) for v in vals[k:]]
    if sum(rest) == 0:
        raise ValueError("residue has zero trace")
    return spectrum_entropy(rest) / math.log(n - k)


def stream(sample, steps, alpha, W=None, snapshot=None, snapshot_at=None):
    """Run W through `steps` observations from sample(t) (t = 1..steps).

    snapshot(t, W) is called when t is in snapshot_at. Returns (W, snaps).
    """
    check_alpha(alpha)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer")
    snaps = []
    at = set(snapshot_at or ())
    for t in range(1, steps + 1):
        x = sample(t)
        if W is None:
            W = zeros(len(x))
        hebbian_update(W, x, alpha)
        if snapshot and t in at:
            snaps.append(snapshot(t, W))
    return W, snaps


def pattern_sampler(patterns, sigma, rng, freqs=None, scales=None):
    patterns = check_patterns(patterns)
    freqs = check_freqs(freqs, len(patterns))
    cum, s = [], 0.0
    for f in freqs:
        s += f
        cum.append(s)

    def sample(_t):
        r = rng.random() * s
        i = next((i for i, c in enumerate(cum) if r < c), len(cum) - 1)
        return noisy(patterns[i], sigma, rng, scales)
    return sample


def noise_sampler(n, rng, scales=None):
    scales = scales or [1.0] * n

    def sample(_t):
        return normalize([s * complex(rng.gauss(0, 1), rng.gauss(0, 1)) for s in scales])
    return sample


def crystallize(patterns, k, steps, sigma, alpha, seed, freqs=None,
                snapshot_at=None, scales=None):
    """Stream noisy observations of `patterns`.

    Returns (W, snaps) with snaps = (t, entropy, subspace_fidelity,
    match_fidelity) at each t in snapshot_at.
    """
    patterns = check_patterns(patterns)
    n = len(patterns[0])
    check_k(k, n)
    rng = random.Random(seed)
    sample = pattern_sampler(patterns, sigma, rng, freqs, scales)

    def snap(t, W):
        vals, vecs = eigh_hermitian(W)
        top = vecs[:k]
        return (t, spectrum_entropy(vals), subspace_fidelity(top, patterns),
                match_fidelity(top, patterns) if k >= len(patterns) else None)
    return stream(sample, steps, alpha, snapshot=snap, snapshot_at=snapshot_at)


def self_recreate(patterns, k, generations, steps, sigma, alpha, freqs=None, seed=0):
    """Generation g+1 learns from noisy observations of generation g's top-k.

    Returns per-generation (match_fidelity, subspace_fidelity) to the
    original patterns.
    """
    true = check_patterns(patterns)
    check_k(k, len(true[0]))
    if k != len(true):
        raise ValueError("self_recreate needs k equal to the number of patterns")
    cur = true
    out = []
    for g in range(generations):
        W, _ = crystallize(cur, k, steps, sigma, alpha, seed=seed * 1000003 + g, freqs=freqs)
        _, vecs = eigh_hermitian(W)
        cur = vecs[:k]
        out.append((match_fidelity(cur, true), subspace_fidelity(cur, true)))
    return out
