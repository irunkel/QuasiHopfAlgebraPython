"""The interface every concrete algebra in this package implements.

A plain Hopf algebra is the special case of a quasi-Hopf algebra where
the associator Phi is trivial (the unit of H^{(x)3}) -- so there is a
single interface, :class:`QuasiHopfAlgebra`, rather than a chain of
"HopfAlgebra extends Bialgebra extends Algebra" classes. Concrete
examples that happen to be honest Hopf algebras can simply return the
trivial associator and alpha=beta=1.

Each concrete subclass is expected to be small and hand-written: it
knows its own generators, relations, and normal-form rewriting (see
e.g. examples/quantum_sl2_quasi.py), computed directly rather than via
any generic Groebner-basis-style machinery. What lives *here* is only
the handful of pieces that are the same for every algebra: the bilinear
extension of the product from basis elements to general elements, and a
default (overridable) coefficient simplification hook.

The axiom checks in :mod:`hopfsym.axioms` are written purely against
this interface, so they work unchanged for any algebra that implements
it -- including, eventually, further generalisations (see CLAUDE.md for
notes on the planned quasi-Hopf group-coalgebra extension).
"""

from __future__ import annotations

from .element import Element


class QuasiHopfAlgebra:
    """Interface for a finite-dimensional quasi-Hopf algebra with a
    known basis, given by hand-written structure maps.

    Required of a subclass:

    - ``basis()``: an iterable of all basis keys (used for exhaustive or
      random axiom sampling).
    - ``unit()``: the multiplicative identity, as an Element.
    - ``multiply_basis(b1, b2)``: the product of two basis elements, as
      an Element. This is where the algebra's defining relations and
      normal-form rewriting live.
    - ``comul(elem)``: the (possibly non-coassociative) coproduct
      Delta, as an algebra homomorphism H -> H (x) H.
    - ``counit(elem)``: the counit epsilon, returning a scalar
      (a QRational).
    - ``antipode(elem)``: the antipode S (an anti-homomorphism).
    - ``associator()`` / ``associator_inv()``: Phi and Phi^-1, elements
      of H^{(x)3}. For an honest Hopf algebra these are just
      ``tensor(tensor(unit, unit), unit)``.
    - ``alpha()`` / ``beta()``: the evaluation/coevaluation elements
      used in the (quasi-)antipode axiom. For an honest Hopf algebra
      these are both just ``unit()``.

    ``mul`` (the bilinear extension of ``multiply_basis``) is provided
    here and should not need to be overridden.
    """

    # -- to be implemented by subclasses -----------------------------
    def basis(self):
        raise NotImplementedError

    def unit(self) -> Element:
        raise NotImplementedError

    def multiply_basis(self, b1, b2) -> Element:
        raise NotImplementedError

    def comul(self, elem: Element) -> Element:
        raise NotImplementedError

    def counit(self, elem: Element):
        raise NotImplementedError

    def antipode(self, elem: Element) -> Element:
        raise NotImplementedError

    def associator(self) -> Element:
        raise NotImplementedError

    def associator_inv(self) -> Element:
        raise NotImplementedError

    def alpha(self) -> Element:
        raise NotImplementedError

    def beta(self) -> Element:
        raise NotImplementedError

    # -- provided for free --------------------------------------------
    def mul(self, a: Element, b: Element) -> Element:
        """The bilinear extension of multiply_basis to general elements."""
        result = Element()
        for k1, c1 in a.terms.items():
            for k2, c2 in b.terms.items():
                prod = self.multiply_basis(k1, k2)
                for k3, c3 in prod.terms.items():
                    result.add_term(k3, c1 * c2 * c3)
        return result
