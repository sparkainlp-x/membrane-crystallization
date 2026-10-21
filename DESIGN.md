# Design notes

SYNTHETIC. Classical numerical experiments on seeded synthetic data only.

## What the update rule is

The draft calls it crystallization. In standard terms:

    W_t = (1 - alpha) W_{t-1} + alpha |x_t><x_t|

is an exponentially weighted running estimate of the observation second-moment (covariance) matrix E[|x><x|]. Its top-K eigenvectors are the principal components of that estimate, so the package does streaming PCA, the setting studied by Oja (1982). It is Hebbian in Hebb's (1949) sense: co-active components strengthen their coupling. "Crystallization" is kept as a descriptive label for the estimate settling, nothing more.

## Forgetting

The draft said that fixed plasticity "forgets nothing". That is not right. An observation seen s steps ago carries weight alpha (1 - alpha)^s, so old data fades exponentially, with an effective window of about 1/alpha steps. After a switch from pattern A to pattern B, B's weight overtakes A's after ln 2 / -ln(1 - alpha) steps (about 0.69/alpha). The sweep measures this crossover on the top eigenvector and finds it within a few steps of that value for alpha = 0.05, 0.02 and 0.01 (results/report.md).

## Learning curve

Snapshots start at step 1. After one step W has rank one, so the spectrum entropy is 0. It rises to about 1.53 nats by step 10 and 1.78 by step 50, and is flat at about 1.81 from step 150 onward (maximum ln 8 = 2.08). The noise floor keeps it below maximum. Subspace fidelity reaches about 0.96 by step 200 and then fluctuates around 0.96 to 0.97. Almost everything happens in the first one or two windows (1/alpha = 100 steps).

## Identifiability

With uniform pattern frequencies, the patterns' eigenvalues coincide, so the top-K eigenspace is identified but its basis is arbitrary. Distinct frequencies (0.6/0.3/0.1 here) separate the eigenvalues and make individual patterns identifiable. Subspace fidelity is therefore the primary metric; per-pattern match fidelity is reported alongside it.

## Residue whiteness

Whiteness is the entropy of the eigenvalues left after removing the top k, divided by log(N - k). With isotropic noise it is close to 1 by construction. The controls show that it does not discriminate the way the draft suggested:

- A pure-noise stream with no patterns scores 0.995 at k = 3, slightly higher than the 3-pattern stream (0.993). The metric cannot tell "all structure extracted" from "there was no structure".
- With k too small, it drops (0.929 at k = 1, 0.978 at k = 2), so it does respond to leftover patterns, but only modestly.
- With coloured (anisotropic) noise it is well below 1 even with the correct k (0.772), and pure coloured noise scores similarly (0.83). It then measures the noise shape, not missing patterns.

Whiteness is reported for reference only and is not used as a completion criterion.

## Self-recreation

Generation g + 1 learns from noisy observations of generation g's top-K eigenvectors, never the originals. Over 20 generations and 50 seeds, fidelity to the original patterns decays steadily at every noise level tested and does not settle within 20 generations. At sigma = 0.1 match fidelity falls from 0.995 to 0.916; at 0.25, from 0.953 to 0.494; at 0.5, from 0.718 to 0.216. For reference, a random 3-dimensional subspace of 8 dimensions has expected subspace fidelity 3/8 = 0.375; at sigma = 0.5 the chain is close to that (0.415 at generation 20). The draft's "infinite horizon" framing is dropped: noise accumulates each generation.

## Eigensolver

eigh_hermitian embeds H in a real symmetric matrix twice its size, where each eigenvalue of H appears twice. The draft kept every other real eigenvector after sorting. With repeated eigenvalues (for example the identity, or uniform pattern frequencies), that can return the same complex vector twice: for the 4 x 4 identity the draft's vectors were not orthonormal and did not reconstruct H (error 1.0). The fix groups equal eigenvalues, maps all real eigenvectors in a group back to complex vectors, keeps an orthonormal set of the right size by Gram-Schmidt, and refines each eigenvalue with the Rayleigh quotient. Tests check orthonormality, reconstruction and eigenspace spans against matrices with known repeated spectra in random bases.

## Limits

- The stream must contain the patterns. Nothing is learned from pure noise.
- Toy scale: 8 dimensions, 3 patterns. Jacobi on the 2N real embedding is O(N^3) per sweep.
- Fixed alpha only. A decaying alpha would trade tracking speed for lower variance; not studied here.
