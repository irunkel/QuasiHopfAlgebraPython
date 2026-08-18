"""Formal linear combinations, and tensor-power bookkeeping, that are
shared by every algebra in this package.

An :class:`Element` is just ``sum_i coeff_i * basis_i``: a dict from
*basis keys* to coefficients. This module has no idea what a basis key
*means* -- that is entirely up to whichever
:class:`~hopfsym.algebra.QuasiHopfAlgebra` created the Element -- and
it is equally agnostic about what ring the coefficients live in: a
coefficient can be a plain ``int``/``Fraction``, a
:class:`hopfsym.qring.QRational` (free q), a
:class:`hopfsym.qring.CycloNum` (q a root of unity), or in principle
anything else that supports ``+``, unary/binary ``-``, ``*``, and
``.is_zero()`` -- see :mod:`hopfsym.qring` for why an algebra might need
one ring or the other. All this module provides is the additive
vector-space structure over whichever ring is in play, plus generic
n-fold tensor-product bookkeeping (building ``H (x) H``,
``H (x) H (x) H``, ... elements, and applying a structure map to one
factor of such a tensor -- used for axioms like the pentagon identity,
which apply the coproduct to one leg of the associator).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple


def _is_zero(coeff) -> bool:
    is_zero = getattr(coeff, "is_zero", None)
    return is_zero() if is_zero is not None else coeff == 0


class TensorKey(tuple):
    """A basis key for an n-fold tensor product: a flat tuple of the
    individual factors' own basis keys.

    This is deliberately a distinct type from a plain tuple (even though
    it behaves like one) so that ``tensor()`` can tell "one basis key
    that happens to be a tuple" apart from "several basis keys tensored
    together", regardless of what an individual algebra uses as its own
    basis key representation.
    """

    def __new__(cls, factors):
        return super().__new__(cls, factors)

    def __repr__(self):
        return " (x) ".join(repr(f) for f in self)


def _factors_of(key) -> tuple:
    """View any basis key as a tuple of tensor factors: a TensorKey is
    already such a tuple; anything else counts as a single factor."""
    return tuple(key) if isinstance(key, TensorKey) else (key,)


def _key_from_factors(factors: tuple):
    """Inverse of _factors_of: a 1-tuple collapses back to a plain key,
    anything longer becomes a TensorKey."""
    return factors[0] if len(factors) == 1 else TensorKey(factors)


class Element:
    """A formal linear combination ``{basis_key: coefficient}``."""

    __slots__ = ("terms",)

    def __init__(self, terms: Dict = None):
        self.terms: Dict = {}
        if terms:
            for k, v in terms.items():
                self.add_term(k, v)

    @classmethod
    def zero(cls) -> "Element":
        return cls()

    @classmethod
    def basis(cls, key, coeff=1) -> "Element":
        return cls({key: coeff})

    def add_term(self, key, coeff) -> None:
        if _is_zero(coeff):
            return
        if key in self.terms:
            new = self.terms[key] + coeff
            if _is_zero(new):
                del self.terms[key]
            else:
                self.terms[key] = new
        else:
            self.terms[key] = coeff

    def __add__(self, other: "Element") -> "Element":
        if not isinstance(other, Element):
            return NotImplemented
        result = Element(dict(self.terms))
        for k, v in other.terms.items():
            result.add_term(k, v)
        return result

    def __neg__(self) -> "Element":
        return Element({k: -v for k, v in self.terms.items()})

    def __sub__(self, other: "Element") -> "Element":
        if not isinstance(other, Element):
            return NotImplemented
        return self + (-other)

    def __rmul__(self, scalar) -> "Element":
        return Element({k: scalar * v for k, v in self.terms.items()})

    def __mul__(self, scalar) -> "Element":
        if isinstance(scalar, Element):
            raise TypeError(
                "Element * Element is ambiguous (algebra product or tensor?); "
                "use alg.mul(a, b) or tensor(a, b) explicitly"
            )
        return self.__rmul__(scalar)

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def __eq__(self, other):
        if not isinstance(other, Element):
            return NotImplemented
        return (self - other).is_zero()

    def __repr__(self):
        if not self.terms:
            return "0"
        return " + ".join(f"({v})*{k!r}" for k, v in self.terms.items())


def tensor(a: Element, b: Element) -> Element:
    """The bilinear tensor product of two elements. Repeated application
    (e.g. ``tensor(tensor(a, b), c)`` or ``tensor(a, tensor(b, c))``)
    flattens automatically into one n-fold TensorKey, regardless of how
    the calls are nested."""
    result = Element()
    for k1, c1 in a.terms.items():
        for k2, c2 in b.terms.items():
            key = _key_from_factors(_factors_of(k1) + _factors_of(k2))
            result.add_term(key, c1 * c2)
    return result


def apply_to_factor(elem: Element, position: int, func: Callable[[Element], Element]) -> Element:
    """Apply the linear map ``func`` to the tensor factor at ``position``
    (0-indexed) of every term of ``elem``, leaving the other factors
    untouched. If ``func`` itself returns a multi-factor tensor element
    (e.g. a coproduct), the arity grows accordingly.

    This is the generic building block behind axiom checks that need
    things like ``(Delta (x) id)(x)`` or ``(id (x) id (x) Delta)(x)``.
    """
    result = Element()
    for key, coeff in elem.terms.items():
        factors = _factors_of(key)
        before, target, after = factors[:position], factors[position], factors[position + 1:]
        mapped = func(Element.basis(target, coeff))
        for mkey, mcoeff in mapped.terms.items():
            mfactors = _factors_of(mkey)
            newfactors = before + mfactors + after
            result.add_term(_key_from_factors(newfactors), mcoeff)
    return result


def tensor_mul(alg, a: Element, b: Element) -> Element:
    """Multiply two elements of ``H^{(x)n}`` componentwise, using
    ``alg.mul`` on each pair of matching factors -- i.e. the algebra
    structure on the n-fold tensor power. Works for any common arity n
    (n=1 is just ``alg.mul(a, b)``)."""
    result = Element()
    for keyA, cA in a.terms.items():
        for keyB, cB in b.terms.items():
            factorsA, factorsB = _factors_of(keyA), _factors_of(keyB)
            if len(factorsA) != len(factorsB):
                raise ValueError(
                    f"tensor_mul: mismatched tensor arity ({len(factorsA)} vs {len(factorsB)})"
                )
            piece = None
            for fa, fb in zip(factorsA, factorsB):
                prod = alg.mul(Element.basis(fa), Element.basis(fb))
                piece = prod if piece is None else tensor(piece, prod)
            result = result + (cA * cB) * piece
    return result
