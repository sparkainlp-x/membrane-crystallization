"""membrane_crystallization: SYNTHETIC streaming PCA with a Hebbian EMA.

W <- (1 - alpha) W + alpha |x><x| is a running estimate of the observation
covariance; its top-K eigenvectors are its principal components (Oja 1982).
"Crystallization" is a descriptive label only. Standard library only.
"""

from .linalg import eigh_hermitian, fidelity, inner, normalize, reconstruct
from .stream import (crystallize, hebbian_update, match_fidelity, noisy,
                     residue_whiteness, self_recreate, spectrum_entropy,
                     stream, subspace_fidelity)

__version__ = "0.1.0"
__all__ = ["eigh_hermitian", "fidelity", "inner", "normalize", "reconstruct",
           "crystallize", "hebbian_update", "match_fidelity", "noisy",
           "residue_whiteness", "self_recreate", "spectrum_entropy", "stream",
           "subspace_fidelity", "__version__"]
