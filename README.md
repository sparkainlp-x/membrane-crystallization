# Membrane Crystallization

**Live demo:** [Holographic Thoughts](https://sparkainlp-x.github.io/holographic-thoughts/) (interactive browser view; synthetic, classical simulation; measured numbers are in this repo)

A small, standard-library Python package for streaming PCA with a Hebbian running average. It learns principal components from a stream of noisy observations and measures how fast it learns, how fast it forgets, and what happens when each generation learns only from the previous one's output.

**SYNTHETIC.** Every number below is measured in this repo on seeded synthetic data (three Bell-state patterns embedded in 8 complex dimensions). Nothing here is hardware or field data.

## What it is

`W <- (1 - alpha) W + alpha |x><x|` is an exponentially weighted running estimate of the observation covariance. Its top-K eigenvectors are its principal components: streaming PCA, as in Oja (1982). "Crystallization" is a descriptive label for the estimate settling, nothing more.

- **It forgets.** With fixed alpha, an observation from s steps ago has weight alpha (1 - alpha)^s, so the memory window is about 1/alpha steps.
- **It recovers the subspace first.** With equal pattern frequencies the eigenvalues coincide and individual patterns are not identifiable; distinct frequencies (0.6/0.3/0.1) separate them.
- **Eigensolver.** `eigh_hermitian` is a stdlib Jacobi solver on the real embedding of a complex Hermitian matrix. It returns a full orthonormal eigenbasis for repeated eigenvalues too, which the draft did not.

## Quick start

```bash
python -m membrane_crystallization --quick        # smoke run, a few seconds
python -m membrane_crystallization --jobs 8       # full tables (results/report.md), about 45 s on 8 cores
pip install -r requirements-dev.txt && python -m pytest   # 58 tests, warnings are errors
```

```python
from membrane_crystallization import crystallize, eigh_hermitian, subspace_fidelity
from membrane_crystallization.sweeps import PATTERNS, FREQS

W, snaps = crystallize(PATTERNS, k=3, steps=3000, sigma=0.25, alpha=0.01, seed=7,
                       freqs=FREQS, snapshot_at=[50, 3000])
vals, vecs = eigh_hermitian(W)
subspace_fidelity(vecs[:3], PATTERNS)
```

Invalid input raises `ValueError`: empty or zero-norm patterns, mismatched lengths, k out of range, frequencies of the wrong length, negative or not summing to 1, alpha outside (0, 1], non-positive steps, negative sigma, non-Hermitian matrices.

## Measured in this repo

Settings unless stated: sigma = 0.25, alpha = 0.01, 6000 steps, frequencies 0.6/0.3/0.1. Values are mean ± sd over seeds.

**Learning curve** (20 seeds; maximum entropy ln 8 = 2.079):

| step | spectrum entropy | subspace fidelity | match fidelity |
|---|---|---|---|
| 1 | 0.000 ± 0.000 | 0.452 ± 0.103 | 0.320 ± 0.100 |
| 10 | 1.531 ± 0.085 | 0.666 ± 0.077 | 0.501 ± 0.067 |
| 25 | 1.724 ± 0.045 | 0.822 ± 0.077 | 0.691 ± 0.078 |
| 50 | 1.775 ± 0.035 | 0.889 ± 0.057 | 0.800 ± 0.065 |
| 100 | 1.797 ± 0.029 | 0.922 ± 0.062 | 0.885 ± 0.058 |
| 150 | 1.803 ± 0.023 | 0.956 ± 0.018 | 0.928 ± 0.020 |
| 200 | 1.805 ± 0.021 | 0.961 ± 0.023 | 0.945 ± 0.027 |
| 250 | 1.807 ± 0.021 | 0.961 ± 0.015 | 0.947 ± 0.016 |
| 300 | 1.808 ± 0.016 | 0.969 ± 0.011 | 0.955 ± 0.016 |
| 350 | 1.808 ± 0.012 | 0.966 ± 0.020 | 0.951 ± 0.020 |
| 400 | 1.810 ± 0.014 | 0.962 ± 0.015 | 0.951 ± 0.016 |
| 450 | 1.810 ± 0.013 | 0.966 ± 0.015 | 0.956 ± 0.016 |
| 500 | 1.810 ± 0.013 | 0.967 ± 0.012 | 0.957 ± 0.014 |
| 1000 | 1.812 ± 0.019 | 0.969 ± 0.012 | 0.956 ± 0.013 |
| 2000 | 1.810 ± 0.013 | 0.963 ± 0.017 | 0.951 ± 0.017 |
| 4000 | 1.812 ± 0.013 | 0.967 ± 0.017 | 0.952 ± 0.020 |
| 6000 | 1.815 ± 0.017 | 0.968 ± 0.015 | 0.958 ± 0.016 |

Entropy is 0 after one step (W has rank one), rises within the first window and is flat at about 1.81 from step 150. Fidelity settles around 0.96 to 0.97 by step 200.

**Forgetting.** Learn pattern A for 20/alpha steps, switch to pattern B, and count steps until the top eigenvector is closer to B than to A (50 seeds; checked every step for alpha = 0.05, every 2 steps for 0.02 and every 5 steps for 0.01, so measured values round up to that resolution). The prediction is the step where B's exponential weight passes A's, ln 2 / -ln(1 - alpha).

| alpha | window 1/alpha | predicted crossover | measured crossover | tracked |
|---|---|---|---|---|
| 0.05 | 20 | 13.5 | 13.6 ± 1.1 | 50/50 |
| 0.02 | 50 | 34.3 | 34.8 ± 2.2 | 50/50 |
| 0.01 | 100 | 69.0 | 72.1 ± 3.2 | 50/50 |

**Residue whiteness controls** (20 seeds). Entropy of the eigenvalues left after removing the top k, over log(N - k).

| stream | k=1 | k=2 | k=3 |
|---|---|---|---|
| 3 patterns, isotropic noise | 0.929 ± 0.009 | 0.978 ± 0.006 | 0.993 ± 0.002 |
| 3 patterns, coloured noise | 0.714 ± 0.010 | 0.716 ± 0.014 | 0.772 ± 0.021 |
| noise | 0.993 ± 0.001 | 0.994 ± 0.001 | 0.995 ± 0.001 |
| coloured noise only | 0.816 ± 0.012 | 0.845 ± 0.013 | 0.832 ± 0.017 |

The metric does not discriminate as a completion criterion: a pure-noise stream scores as white as the 3-pattern stream, and coloured noise lowers it even with the right k. It does fall when k is too small, but only modestly. It is reported for reference only.

**Self-recreation** (50 seeds, 20 generations). Each generation learns only from noisy observations of the previous generation's top-3 eigenvectors. Match fidelity to the original patterns:

| sigma | gen 1 | gen 2 | gen 3 | gen 5 | gen 10 | gen 15 | gen 20 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.995 ± 0.002 | 0.991 ± 0.003 | 0.987 ± 0.005 | 0.977 ± 0.008 | 0.957 ± 0.015 | 0.934 ± 0.021 | 0.916 ± 0.024 |
| 0.25 | 0.953 ± 0.016 | 0.910 ± 0.029 | 0.871 ± 0.036 | 0.798 ± 0.058 | 0.674 ± 0.067 | 0.566 ± 0.087 | 0.494 ± 0.087 |
| 0.5 | 0.718 ± 0.085 | 0.553 ± 0.079 | 0.479 ± 0.104 | 0.396 ± 0.089 | 0.289 ± 0.080 | 0.233 ± 0.068 | 0.216 ± 0.071 |

Subspace fidelity (chance level for a random 3-dimensional subspace of 8 dimensions: 0.375):

| sigma | gen 1 | gen 2 | gen 3 | gen 5 | gen 10 | gen 15 | gen 20 |
|---|---|---|---|---|---|---|---|
| 0.1 | 0.997 ± 0.001 | 0.995 ± 0.002 | 0.992 ± 0.003 | 0.987 ± 0.005 | 0.974 ± 0.010 | 0.960 ± 0.012 | 0.947 ± 0.014 |
| 0.25 | 0.968 ± 0.013 | 0.940 ± 0.021 | 0.913 ± 0.029 | 0.866 ± 0.040 | 0.780 ± 0.050 | 0.701 ± 0.062 | 0.652 ± 0.073 |
| 0.5 | 0.807 ± 0.072 | 0.680 ± 0.048 | 0.640 ± 0.060 | 0.583 ± 0.068 | 0.485 ± 0.065 | 0.440 ± 0.067 | 0.415 ± 0.074 |

Fidelity decays at every noise level and does not settle within 20 generations. There is no infinite horizon here: noise accumulates each generation.

## Limits

- SYNTHETIC toy scale: 8 dimensions, 3 patterns, seeded noise.
- Nothing is learned from a stream without patterns.
- Fixed alpha only; decaying schedules are not studied.
- The Jacobi solver is O(N^3) per sweep on a 2N embedding; fine here, slow for large N.

See [DESIGN.md](DESIGN.md) for the reasoning and the eigensolver fix.

## Provenance

The package as received is kept unchanged in [spec/](spec/). [CHANGELOG.md](CHANGELOG.md) lists what changed.

## Acknowledgements

Thanks to Mun Seok Lee, who is credited on the original package kept in [spec/](spec/). The acknowledgement does not imply his endorsement of this code or its conclusions.

## References

- Hebb, D. O. (1949). *The Organization of Behavior: A Neuropsychological Theory.* Wiley.
- Oja, E. (1982). Simplified neuron model as a principal component analyzer. *Journal of Mathematical Biology* 15(3), 267-273. https://doi.org/10.1007/BF00275687

## How to cite

See [CITATION.cff](CITATION.cff).

## License

Copyright (c) 2026 Jean-François Brisson, Spark AI NLP.

This software is available under the GNU Affero General Public License v3.0 only (AGPL-3.0-only); see [LICENSE](LICENSE).
