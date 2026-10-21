# Membrane Crystallization

**A holographic membrane that learns patterns from a noisy stream —
and re-derives itself, generation after generation.**

Online Hebbian integration of noisy observations; crystallization
measured three ways (spectrum entropy, extraction fidelity, residue
whiteness); self-recreation chains where each generation learns
from the previous one's extracted patterns. Stdlib-only Python
(includes a cyclic Jacobi eigensolver for Hermitian matrices).
AGPL-3.0.

## Verified

- Subspace fidelity 0.96–0.97 from a noisy stream (σ=0.25)
- Residue whiteness 0.988 — all structure extracted, noise left
  behind
- Self-recreation: 0.955 → 0.820 over five generations (graceful
  decay — the measured horizon of "infinite")
- 7/7 tests pass

## Run it

```bash
python tests.py   # 7 tests
python demo.py    # crystallization, whiteness, recreation chain
```

## Honest limits

Needs the stream to contain the patterns (no bootstrap from pure
noise); Hebbian learning recovers the subspace, not a basis —
distinct frequencies make patterns identifiable; "infinite" has a
measured horizon; toy scale. Full account in `DESIGN.md`.
