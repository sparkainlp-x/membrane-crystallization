"""Seeded sweeps behind the README tables. SYNTHETIC data only."""

import math
import random
import statistics
from concurrent.futures import ProcessPoolExecutor

from .linalg import eigh_hermitian, fidelity
from .stream import (crystallize, noise_sampler, pattern_sampler,
                     residue_whiteness, self_recreate, spectrum_entropy,
                     stream, subspace_fidelity)

S2 = 2 ** -0.5
N = 8
PATTERNS = [[S2, 0, 0, S2, 0, 0, 0, 0],      # Bell states embedded in 8 dims
            [S2, 0, 0, -S2, 0, 0, 0, 0],
            [0, S2, S2, 0, 0, 0, 0, 0]]
FREQS = [0.6, 0.3, 0.1]
SIGMA, ALPHA, STEPS = 0.25, 0.01, 6000


def _mean_sd(xs):
    return (statistics.fmean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def _pmap(fn, args, jobs):
    if jobs and jobs > 1:
        with ProcessPoolExecutor(jobs) as ex:
            return list(ex.map(fn, args))
    return [fn(a) for a in args]


# ---------------------------------------------------------------- curve
CURVE_T = [1, 10, 25] + list(range(50, 501, 50)) + [1000, 2000, 4000, 6000]


def _curve_one(seed):
    _, snaps = crystallize(PATTERNS, 3, STEPS, SIGMA, ALPHA, seed, FREQS, snapshot_at=CURVE_T)
    return snaps


def entropy_curve(seeds=20, jobs=1):
    runs = _pmap(_curve_one, range(seeds), jobs)
    rows = []
    for i, t in enumerate(CURVE_T):
        e = _mean_sd([r[i][1] for r in runs])
        s = _mean_sd([r[i][2] for r in runs])
        m = _mean_sd([r[i][3] for r in runs])
        rows.append({"t": t, "entropy": e, "subspace": s, "match": m})
    return {"rows": rows, "max_entropy": math.log(N), "seeds": seeds}


# ---------------------------------------------------------------- forgetting
def _switch_one(args):
    seed, alpha = args
    rng = random.Random(seed)
    a, b = [PATTERNS[0]], [PATTERNS[1]]
    sa, sb = pattern_sampler(a, SIGMA, rng), pattern_sampler(b, SIGMA, rng)
    burn = int(round(20 / alpha))
    W, _ = stream(sa, burn, alpha)
    cross = None
    t = 0
    while t < int(20 / alpha):
        t += 1
        W, _ = stream(sb, 1, alpha, W=W)
        _, vecs = eigh_hermitian(W) if (t % max(1, int(0.05 / alpha)) == 0) else (None, None)
        if vecs is not None and fidelity(vecs[0], b[0]) > fidelity(vecs[0], a[0]):
            cross = t
            break
    return cross


def forgetting(alphas=(0.05, 0.02, 0.01), seeds=50, jobs=1):
    """Learn pattern A, switch to pattern B; steps until the top eigenvector
    is closer to B than to A. Weighting predicts about ln(2)/alpha."""
    rows = []
    for alpha in alphas:
        res = _pmap(_switch_one, [(s, alpha) for s in range(seeds)], jobs)
        ok = [r for r in res if r is not None]
        rows.append({"alpha": alpha, "window": 1 / alpha,
                     "predicted": math.log(2) / math.log(1 / (1 - alpha)),
                     "measured": _mean_sd(ok), "tracked": len(ok), "seeds": seeds,
                     "resolution": max(1, int(0.05 / alpha))})
    return rows


# ---------------------------------------------------------------- whiteness
def _white_one(args):
    case, seed = args
    rng = random.Random(seed)
    if case == "noise":
        W, _ = stream(noise_sampler(N, rng), STEPS, ALPHA)
    elif case == "coloured":
        scales = [2.0, 1.5, 1.0, 1.0, 0.7, 0.5, 0.4, 0.3]
        W, _ = stream(pattern_sampler(PATTERNS, SIGMA, rng, FREQS, scales), STEPS, ALPHA)
    elif case == "coloured noise only":
        scales = [2.0, 1.5, 1.0, 1.0, 0.7, 0.5, 0.4, 0.3]
        W, _ = stream(noise_sampler(N, rng, scales), STEPS, ALPHA)
    else:
        W, _ = stream(pattern_sampler(PATTERNS, SIGMA, rng, FREQS), STEPS, ALPHA)
    return [residue_whiteness(W, k) for k in (1, 2, 3)]


WHITE_CASES = ["3 patterns, isotropic noise", "3 patterns, coloured noise",
               "noise", "coloured noise only"]


def whiteness_controls(seeds=20, jobs=1):
    out = {}
    for case in WHITE_CASES:
        key = {"3 patterns, isotropic noise": "iso", "3 patterns, coloured noise": "coloured"}.get(case, case)
        res = _pmap(_white_one, [(key, s) for s in range(seeds)], jobs)
        out[case] = {k: _mean_sd([r[k - 1] for r in res]) for k in (1, 2, 3)}
    return out


# ---------------------------------------------------------------- recreation
RECREATE_SIGMAS = (0.1, 0.25, 0.5)


def _recreate_one(args):
    seed, sigma, gens, steps = args
    return self_recreate(PATTERNS, 3, gens, steps, sigma, ALPHA, FREQS, seed=seed)


def recreation(seeds=50, sigmas=RECREATE_SIGMAS, generations=20, steps=STEPS, jobs=1):
    out = {}
    for sigma in sigmas:
        res = _pmap(_recreate_one, [(s, sigma, generations, steps) for s in range(seeds)], jobs)
        out[sigma] = [{"gen": g + 1,
                       "match": _mean_sd([r[g][0] for r in res]),
                       "subspace": _mean_sd([r[g][1] for r in res])}
                      for g in range(generations)]
    return {"seeds": seeds, "steps": steps, "by_sigma": out}


# ---------------------------------------------------------------- report
def _ms(x):
    return f"{x[0]:.3f} ± {x[1]:.3f}"


def report(curve_seeds=20, switch_seeds=50, white_seeds=20, rec_seeds=50,
           generations=20, jobs=1):
    L = []
    c = entropy_curve(curve_seeds, jobs)
    L.append(f"Learning curve (3 patterns, freqs {FREQS}, sigma={SIGMA}, alpha={ALPHA}; "
             f"mean ± sd over {curve_seeds} seeds; max entropy ln 8 = {c['max_entropy']:.3f})")
    L.append("| step | spectrum entropy | subspace fidelity | match fidelity |")
    L.append("|---|---|---|---|")
    for r in c["rows"]:
        L.append(f"| {r['t']} | {_ms(r['entropy'])} | {_ms(r['subspace'])} | {_ms(r['match'])} |")
    L.append(f"\nForgetting: learn pattern A, switch to B ({switch_seeds} seeds)")
    L.append("| alpha | window 1/alpha | predicted crossover | measured crossover | tracked |")
    L.append("|---|---|---|---|---|")
    for r in forgetting(seeds=switch_seeds, jobs=jobs):
        L.append(f"| {r['alpha']} | {r['window']:.0f} | {r['predicted']:.1f} | "
                 f"{r['measured'][0]:.1f} ± {r['measured'][1]:.1f} | {r['tracked']}/{r['seeds']} |")
    L.append(f"\nResidue whiteness controls (mean ± sd over {white_seeds} seeds)")
    L.append("| stream | k=1 | k=2 | k=3 |")
    L.append("|---|---|---|---|")
    for case, d in whiteness_controls(white_seeds, jobs).items():
        L.append(f"| {case} | {_ms(d[1])} | {_ms(d[2])} | {_ms(d[3])} |")
    r = recreation(rec_seeds, generations=generations, jobs=jobs)
    L.append(f"\nSelf-recreation (match fidelity to the original patterns, mean ± sd over {rec_seeds} seeds)")
    gens_show = [g for g in (1, 2, 3, 5, 10, 15, 20) if g <= generations]
    L.append("| sigma | " + " | ".join(f"gen {g}" for g in gens_show) + " |")
    L.append("|---|" + "---|" * len(gens_show))
    for sigma, rows in r["by_sigma"].items():
        L.append(f"| {sigma} | " + " | ".join(_ms(rows[g - 1]["match"]) for g in gens_show) + " |")
    L.append("subspace fidelity:")
    for sigma, rows in r["by_sigma"].items():
        L.append(f"| {sigma} | " + " | ".join(_ms(rows[g - 1]["subspace"]) for g in gens_show) + " |")
    return "\n".join(L)
