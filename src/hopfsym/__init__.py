"""hopfsym: a small, dependency-free framework for symbolic manipulation
and axiom-checking of (quasi-)Hopf algebras.

See CLAUDE.md for the architecture overview and README.md for a usage
example.
"""

from .element import Element, TensorKey, tensor, apply_to_factor, tensor_mul, Δ, ε
from .algebra import QuasiHopfAlgebra

__all__ = [
    "Element",
    "TensorKey",
    "tensor",
    "apply_to_factor",
    "tensor_mul",
    "Δ",
    "ε",
    "QuasiHopfAlgebra",
]
