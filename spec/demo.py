"""Demo: crystallization and self-recreation.

Patterns: 3 Bell states embedded in 8 dimensions, observed with
frequencies 0.5 / 0.3 / 0.2 (distinct frequencies separate the
eigenvalues, making individual patterns identifiable).

1. Crystallization: stream 4000 noisy observations; watch spectrum
   entropy, subspace fidelity and per-pattern match fidelity.
2. Residue whiteness: after removing the top-3, is the leftover white?
3. Self-recreation: 5 generations, each crystallized from the
   previous generation's extracted patterns; fidelity to the true
   patterns per generation -- the measured horizon of "infinite".

Run: python demo.py
"""

from crystal import (crystallize, eigh_hermitian, spectrum_entropy,
                     match_fidelity, subspace_fidelity,
                     residue_whiteness, self_recreate)

S2 = 2 ** -0.5
BELLS = [[S2, 0j, 0j, S2], [S2, 0j, 0j, -S2], [0j, S2, S2, 0j]]
PATTERNS = [b + [0j] * 4 for b in BELLS]  # embedded in 8 dims
FREQS = [0.6, 0.3, 0.1]
K, SIGMA, ALPHA, STEPS = 3, 0.25, 0.01, 6000

print("== 1. crystallization from a noisy stream ==")
W, snaps = crystallize(PATTERNS, K, STEPS, SIGMA, ALPHA, seed=7,
                       freqs=FREQS, snapshot_every=1000)
print(f"  {'t':>5} {'entropy':>8} {'subspace-F':>10} {'match-F':>8}")
for t, s, fs, fm in snaps:
    print(f"  {t:>5} {s:>8.4f} {fs:>10.4f} {fm:>8.4f}")

print("\n== 2. residue whiteness ==")
w = residue_whiteness(W, K)
print(f"  residue entropy / max = {w:.4f}  (1.0 = perfectly white)")

print("\n== 3. self-recreation chain ==")
fids = self_recreate(PATTERNS, K, 5, STEPS, SIGMA, ALPHA, FREQS, seed=7)
for g, f in enumerate(fids):
    print(f"  generation {g}: fidelity to true patterns = {f:.4f}")

_, _, fs_last, fm_last = snaps[-1]
ok = (fs_last > 0.95 and fm_last > 0.90 and w > 0.95 and
      fids[0] > 0.90 and fids[1] > 0.85)
print("\nRESULT:", "PASS -- subspace crystallizes from noise, residue is "
      "white, self-recreation holds for generations"
      if ok else "FAIL")
raise SystemExit(0 if ok else 1)
