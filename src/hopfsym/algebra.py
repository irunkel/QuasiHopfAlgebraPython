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

from .element import Element, apply_to_factor, flip, tensor, tensor_mul


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

    Optional (only for algebras that are additionally quasi-triangular /
    ribbon -- most are not, so these default to ``NotImplementedError``
    rather than being abstract):

    - ``r_matrix()``: the universal R-matrix, an element of H (x) H,
      making the algebra quasi-triangular (see
      ``axioms.check_r_matrix_intertwiner``, ``axioms.check_hexagon``).
    - ``ribbon()``: the ribbon element, an element of H, extending a
      quasi-triangular structure to a ribbon one (see
      ``axioms.check_ribbon``).
    - ``r_matrix_inv()``/``ribbon_inv()``: the inverses of the above, if
      an algebra's own presentation gives them directly (see
      ``axioms.check_r_matrix_inverse``/``axioms.check_ribbon_inverse``,
      which check them against ``r_matrix()``/``ribbon()``). Unlike
      ``monodromy``/``drinfeld``/``f_element`` below, these are *not*
      generic -- an inverse isn't computable from the interface alone in
      general -- so they stay optional, per-algebra, like ``r_matrix()``/
      ``ribbon()`` themselves.

    ``mul`` (the bilinear extension of ``multiply_basis``), ``elt`` and
    ``tag`` (see their own docstrings) are provided here and should not
    need to be overridden. So are three more structure maps that turn
    out to be *provably generic* -- expressible purely in terms of the
    interface above, with no algebra-specific data at all, unlike
    ``multiply_basis``/``comul``/``antipode``/``r_matrix``/``ribbon``
    (which genuinely differ per algebra and stay hand-written per
    example):

    - ``monodromy()``: M = R_21 . R, an element of H (x) H. Needs only
      ``r_matrix()``.
    - ``drinfeld()``: the Drinfeld element, an element of H, built from
      Phi, R, S, alpha and beta (see ``axioms.check_ribbon``, which
      checks it against the ribbon element).
    - ``f_element()``: the element F relating Delta and S (Drinfeld's
      paper), built from Phi, Delta, S, alpha and beta -- no R-matrix
      needed (see ``axioms.check_s_delta_compatibility``).

    Each of these three raises ``NotImplementedError`` naturally if a
    prerequisite (typically ``r_matrix()``) isn't implemented, and each
    is cached (a lazily-created instance attribute, so no ``__init__``
    cooperation is needed from subclasses) -- computed once per algebra
    instance. If an algebra's own presentation happens to give a
    *simpler* closed form for one of these (e.g. because its associator
    is trivial, or a paper it's ported from states one explicitly),
    that's worth adding as a regression test comparing it against these
    generic implementations, not reimplementing to override them -- see
    e.g. ``tests/test_restricted_sl2.py``'s
    ``test_drinfeld_matches_classical_hopf_formula``/
    ``test_f_element_matches_trivial_formula``, or
    ``tests/test_symplectic_fermion.py``'s explicit-formula checks.
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

    # -- optional: quasi-triangular / ribbon structure -----------------
    def r_matrix(self) -> Element:
        raise NotImplementedError

    def ribbon(self) -> Element:
        raise NotImplementedError

    def r_matrix_inv(self) -> Element:
        raise NotImplementedError

    def ribbon_inv(self) -> Element:
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
        result.alg = self
        return result

    def elt(self, key, coeff=1) -> Element:
        """A basis Element tagged with this algebra: ``Element.basis(key,
        coeff, alg=self)``. Tagged Elements let ``*`` mean this algebra's
        product instead of raising -- e.g. ``E = alg.elt((1,0,0));
        F = alg.elt((0,1,0)); E * F`` is ``alg.mul(E, F)`` (see
        ``Element.__mul__``). ``*`` never means the tensor product; use
        ``tensor(a, b)`` explicitly for that."""
        return Element.basis(key, coeff, alg=self)

    def tag(self, elem: Element) -> Element:
        """A copy of ``elem`` tagged with this algebra, for an Element you
        got some way other than ``elt()``/``mul()``/``Δ`` -- typically one
        of the nullary structure elements (``alg.associator()``,
        ``alg.r_matrix()``, ``alg.ribbon()``, ...), which come back
        untagged since their own implementations build them from
        untagged pieces. A *copy*, not an in-place tag, because several
        of those are cached per-instance (``r_matrix``/``ribbon``/
        ``drinfeld``/``f_element``): tagging the cached object itself
        would leak the tag into every other caller that fetches the same
        cached value. Lets you write e.g. ``Phi = alg.tag(alg.associator())``
        once and then use ``*``/``Δ`` on ``Phi`` freely."""
        result = Element(dict(elem.terms))
        result.alg = self
        return result

    def monodromy(self) -> Element:
        """The monodromy matrix M = R_21 . R, an element of H (x) H.
        Needs only ``r_matrix()`` -- raises ``NotImplementedError``
        (propagated from that call) if it isn't implemented. Cached (a
        lazily-created instance attribute -- no ``__init__`` cooperation
        needed from subclasses), so only computed once per instance."""
        cached = getattr(self, "_monodromy_cache", None)
        if cached is not None:
            return cached
        result = tensor_mul(self, flip(self.r_matrix()), self.r_matrix())
        self._monodromy_cache = result
        return result

    def drinfeld(self) -> Element:
        r"""The Drinfeld element, an element of H:

            u = sum S(p2 . beta . S(p3)) . S(r2) . alpha . r1 . p1

        summed over terms Phi = sum p1 (x) p2 (x) p3 and R = sum r1 (x) r2
        (i.e. over Phi (x) R, a 5-fold tensor) -- the standard formula for
        a quasi-triangular quasi-Hopf algebra (Drinfeld's paper), needing
        only ``associator()``, ``r_matrix()``, ``antipode()``, ``alpha()``
        and ``beta()``. Cached, see ``monodromy()``.

        If an algebra's associator happens to be trivial, this collapses
        to the classical Hopf-algebra formula ``u = sum S(r2) . r1`` --
        worth checking as a regression test in that case (see
        ``tests/test_restricted_sl2.py``) rather than reimplementing the
        simplification here.
        """
        cached = getattr(self, "_drinfeld_cache", None)
        if cached is not None:
            return cached

        alpha, beta = self.alpha(), self.beta()
        result = Element()
        for key, c in tensor(self.associator(), self.r_matrix()).terms.items():
            p1, p2, p3, r1, r2 = key
            inner = self.mul(self.mul(Element.basis(p2), beta), self.antipode(Element.basis(p3)))
            term = self.antipode(inner)
            for factor in (self.antipode(Element.basis(r2)), alpha, Element.basis(r1), Element.basis(p1)):
                term = self.mul(term, factor)
            for k2, c2 in term.terms.items():
                result.add_term(k2, c * c2)

        self._drinfeld_cache = result
        return result

    def f_element(self) -> Element:
        r"""The element F relating Delta and S (Drinfeld's paper), used in
        the S-Delta compatibility check
        ``axioms.check_s_delta_compatibility``. Needs only
        ``associator()``/``associator_inv()``, ``comul()``, ``antipode()``,
        ``alpha()`` and ``beta()`` -- notably *not* ``r_matrix()``, unlike
        ``monodromy()``/``drinfeld()``. Cached, see ``monodromy()``.

        If an algebra's associator happens to be trivial, this collapses
        to the trivial ``F = 1 (x) 1`` -- worth checking as a regression
        test in that case (see ``tests/test_restricted_sl2.py``) rather
        than reimplementing the simplification here.
        """
        cached = getattr(self, "_f_element_cache", None)
        if cached is not None:
            return cached

        Phi = self.associator()
        Phiinv = self.associator_inv()
        alpha, beta = self.alpha(), self.beta()

        xx = tensor_mul(self, tensor(self.unit(), Phi), apply_to_factor(Phiinv, 2, self.comul))

        ga = Element()
        for key, c in xx.terms.items():
            a1, a2, a3, a4 = key
            left = self.mul(self.mul(self.antipode(Element.basis(a2)), alpha), Element.basis(a3))
            right = self.mul(self.mul(self.antipode(Element.basis(a1)), alpha), Element.basis(a4))
            for k2, c2 in tensor(left, right).terms.items():
                ga.add_term(k2, c * c2)

        result = Element()
        for key, c in Phi.terms.items():
            a1, a2, a3 = key
            part1 = Element()
            for k, cc in self.comul(Element.basis(a1)).terms.items():
                u, v = k
                for k2, c2 in tensor(self.antipode(Element.basis(v)), self.antipode(Element.basis(u))).terms.items():
                    part1.add_term(k2, cc * c2)
            inner = self.mul(self.mul(Element.basis(a2), beta), self.antipode(Element.basis(a3)))
            part3 = self.comul(inner)
            term = tensor_mul(self, tensor_mul(self, part1, ga), part3)
            for k2, c2 in term.terms.items():
                result.add_term(k2, c * c2)

        self._f_element_cache = result
        return result
