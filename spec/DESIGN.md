# DESIGN — Deep think: crystallization and self-recreation

## The brief, from October 20

"Infinite self-recreation when crystallization optimizes entropy."
The yin-yang membranes delivered the loop; the learning half was
never built. This is it.

## What was built

No clean patterns are handed to the membrane. It watches a stream
of noisy observations and integrates them Hebbian-style,
W ← (1−α)W + α|x⟩⟨x|. Crystallization is measured three ways:

1. **Spectrum entropy** S(t) of W (von Neumann, normalized).
2. **Extraction fidelity** — top-K eigenvectors vs the true
   patterns, sampled over time. Two numbers: *subspace* fidelity
   (the principled metric: what Hebbian learning actually recovers
   is the span, not a basis) and per-pattern *match* fidelity.
3. **Residue whiteness** — after removing the top-K components,
   the residue's entropy per unit trace as a fraction of maximum.
   ~1.0 means the membrane extracted *all* the structure and left
   white noise: crystallization complete.

**Self-recreation:** generation g+1 crystallizes from noisy
observations of generation g's *extracted* patterns — never the
originals. Fidelity to the true patterns per generation measures
the actual horizon of "infinite," not the slogan.

## Two things the measurements taught us

1. **Hebbian learning recovers the subspace, not the patterns.**
   With uniform pattern frequencies the top-K eigenspace is
   correct but its basis is arbitrary (degeneracy) — individual
   patterns are unidentifiable from W alone. Distinct observation
   frequencies (0.6/0.3/0.1) separate the eigenvalues and make
   patterns identifiable. The code reports subspace fidelity as
   primary for exactly this reason.
2. **The naive "entropy falls" story is wrong.** Spectrum entropy
   rises then plateaus — the noise floor keeps it mixed. What
   "optimizes" is the *residue*: the membrane compresses all
   structure into K eigenvectors and leaves white noise behind
   (whiteness 0.988). Crystallization isn't entropy destruction;
   it's structure extraction.

## Measured

- Subspace fidelity 0.96–0.97 stable from t=1000; match 0.95.
- Residue whiteness 0.988.
- Self-recreation: 0.955 → 0.936 → 0.849 → 0.843 → 0.820 over
  five generations. Graceful decay — "infinite" has a measured
  horizon, and here it is.

## Honest limits

- **Needs the stream to contain the patterns.** No bootstrap from
  pure noise — the same limit as the yin-yang work. A seed of
  structure is required; the membrane concentrates it, it doesn't
  invent it.
- **Fixed plasticity forgets nothing and learns slowly**;
  decaying α would track non-stationary streams but wasn't
  studied here.
- **Toy scale** (8 oscillators, 3 patterns). The Jacobi solver is
  O(N³) — fine here, not the point.
- **"Infinite" is asymptotic.** Five generations measured,
  decaying. The horizon, not the slogan, is the result.
