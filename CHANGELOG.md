# Changelog

## Unreleased

- Documentation only: added a Provenance statement that says which files are derived from the package received under Mun Seok Lee's name and by how much, that he is not an author of this release and has not endorsed it.

## 0.1.0 (2026-10-22)

- Archived on Zenodo: version DOI [10.5281/zenodo.23583050](https://doi.org/10.5281/zenodo.23583050), concept DOI [10.5281/zenodo.23583049](https://doi.org/10.5281/zenodo.23583049).
- First commit: the package exactly as received (`spec/`).
- Authorship: Jean-François Brisson. Mun Seok Lee is credited in the acknowledgements only.
- Naming: the update rule is described as what it is, an exponentially weighted running estimate of the observation covariance whose top-K eigenvectors are its principal components (streaming PCA, Oja 1982). "Crystallization" stays as a descriptive label.
- Fixed: the claim that fixed plasticity "forgets nothing". Old observations decay as (1 - alpha)^s, a window of about 1/alpha steps. A pattern-switch sweep measures the crossover against the ln 2 / -ln(1 - alpha) prediction.
- Fixed: `eigh_hermitian` returned duplicate eigenvectors for repeated eigenvalues (for the 4 x 4 identity, reconstruction error 1.0). It now groups equal eigenvalues, recovers an orthonormal basis by Gram-Schmidt and refines eigenvalues by Rayleigh quotient. Tested on known repeated spectra in random bases.
- Learning curve snapshots now start at step 1. Entropy saturates within about 150 steps; the draft's 1000-step snapshots missed the rise.
- Residue whiteness: added controls (k too small, coloured noise, pure noise with and without colour). Pure noise scores as white as the 3-pattern stream, so whiteness is reported for reference only, not as a completion criterion.
- Self-recreation: 50 seeds, three noise levels, 20 generations. Fidelity decays steadily and does not settle; the "infinite horizon" framing is dropped.
- Input validation for zero-norm or mismatched patterns, k bounds, frequencies (length, sign, sum to 1), alpha, steps, sigma and non-Hermitian matrices.
- "Verified" headings replaced by measurements.
- Added a package layout, a CLI that prints the sweeps, the full report in `results/report.md`, 58 pytest tests, CI on Python 3.10 to 3.13, README, CITATION.cff and .zenodo.json.
