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

from .element import Element, apply_to_factor, flip, tensor, Δ


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
      these are both just ``unit()``. By convention (see
      ``axioms.check_alpha_beta_normalization``, part of ``check_all``)
      these are additionally normalised so ``eps(alpha) == 1 ==
      eps(beta)`` individually -- the axioms alone only force the
      *product* to be 1, but some formulas (e.g. the Drinfeld element
      in the symplectic-fermion paper, arXiv:1706.08164) rely on each
      factor separately, so every algebra here is written to satisfy
      it.

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
    - ``left_integral()`` / ``right_integral()``: an element Lambda of H
      with ``h . Lambda == eps(h) . Lambda`` for all h in H (right:
      ``Lambda . h == eps(h) . Lambda``) -- see
      ``axioms.check_left_integral``/``axioms.check_right_integral`` (the
      latter defined by literally multiplying the other way round, not
      via a separate H^op algebra). Such an element is unique up to
      scalar (when it exists), so the choice of normalisation is up to
      the implementing algebra.
    - ``left_cointegral(elem)`` / ``right_cointegral(elem)``: the dual
      notion, an element of H^* -- so, like ``counit``, represented as a
      method taking an ``Element`` of H and returning a scalar rather
      than as an ``Element`` itself. Checked against their defining
      property (Bulacu-Caenepeel, via arXiv:1812.10445's Definition 3.5)
      by ``axioms.check_left_cointegral``/``axioms.check_right_cointegral``,
      which need ``U()``/``V()`` (resp. ``Ucop()``/``Vcop()``) and
      ``modulus()`` below.
    - ``modulus(elem)``: the algebra morphism gamma: H -> field with
      ``Lambda . h == gamma(h) . Lambda`` for a left integral Lambda
      (H is *unimodular* iff gamma == eps, in which case left and right
      integrals -- and cointegrals -- coincide). This is technically
      derivable from ``left_integral()`` (by finding the proportionality
      constant of ``Lambda . h`` against ``Lambda``, for each basis h --
      guaranteed to exist and be unique by the one-dimensionality of the
      space of left integrals), but is given explicitly here rather than
      computed generically -- cross-check the two against each other in
      a test instead if both are available (see
      ``tests/test_symplectic_fermion.py``).
    - ``antipode_inv(elem)``: the inverse of the antipode S (always
      invertible for a finite-dimensional quasi-Hopf algebra, but not
      computable from the required interface alone in general -- like
      ``r_matrix_inv``/``ribbon_inv``, this is per-algebra, filled in
      only when a source gives it directly). Needed for
      ``qR()``/``pL()``/``U()``/``V()``/``Ucop()``/``Vcop()`` below
      (unlike its ``f_element_inv()`` counterpart, ``pR()``/``qL()``
      don't need it either).

    ``mul`` (the bilinear extension of ``multiply_basis``), ``elt`` and
    ``tag`` (see their own docstrings) are provided here and should not
    need to be overridden. So are five more structure maps that turn
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
    - ``gamma()``: an auxiliary element of H (x) H (Drinfeld's paper, eq
      (1.24)), built from Phi, Phi^-1, S and alpha -- used to build
      ``f_element()``/``f_element_inv()`` below (see
      ``axioms.check_gamma_definition``, eq (1.35)).
    - ``f_element()``: the element F relating Delta and S (Drinfeld's
      paper), built from Phi, Delta, S, alpha and beta -- no R-matrix
      needed (see ``axioms.check_s_delta_compatibility``).
    - ``f_element_inv()``: the inverse of ``f_element()`` (Drinfeld's
      paper again, eq (1.36)) -- needs only the same data as
      ``f_element()`` itself, notably *not* ``antipode_inv()`` (unlike
      most of the cointegral machinery below, which does). No dedicated
      ``axioms.py`` check for it: Drinfeld's eq (1.34),
      ``F.Delta(S(a)).F^-1 == (S (x) S)(Delta'(a))``, is the algebraic
      consequence of ``axioms.check_s_delta_compatibility`` right-
      multiplied by F^-1, so it adds nothing once that check and
      ``F.F^-1 == 1 (x) 1 == F^-1.F`` both hold (the latter verified
      directly per-algebra, e.g.
      ``tests/test_symplectic_fermion.py``'s
      ``test_f_element_inv_is_genuine_inverse``).

    Each of these five raises ``NotImplementedError`` naturally if a
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
    ``tests/test_symplectic_fermion.py``'s explicit-formula checks
    (including, now, ``f_element_inv()`` itself: it was originally
    written by hand for ``SymplecticFermionQ`` before this generic
    version existed, confirmed to agree with it exactly, and kept as a
    regression test rather than an override).

    A second batch of provably generic elements, needed for the
    cointegral defining property (Bulacu-Caenepeel, via
    arXiv:1812.10445's Definition 3.5) -- these additionally need
    ``antipode_inv()`` (optional), so raise ``NotImplementedError`` if
    it isn't implemented:

    - ``qR()``/``pR()``/``qL()``/``pL()``: auxiliary elements of
      H (x) H (eq (3.24) in arXiv:1812.10445), built from Phi, Phi^-1,
      S, S^-1, alpha and beta.
    - ``U()``/``V()``/``Ucop()``/``Vcop()``: built from those plus
      ``f_element()``/``f_element_inv()`` (eq (3.25), (3.28)) -- used by
      ``axioms.check_left_cointegral``/``axioms.check_right_cointegral``.

    Same caching convention as ``monodromy``/``drinfeld``/``f_element``.
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

    # -- optional: integrals and cointegrals ---------------------------
    def left_integral(self) -> Element:
        raise NotImplementedError

    def right_integral(self) -> Element:
        raise NotImplementedError

    def left_cointegral(self, elem: Element):
        raise NotImplementedError

    def right_cointegral(self, elem: Element):
        raise NotImplementedError

    def modulus(self, elem: Element):
        raise NotImplementedError

    def antipode_inv(self, elem: Element) -> Element:
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
        R = self.tag(self.r_matrix())
        result = flip(R) * R
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

        alpha, beta = self.tag(self.alpha()), self.tag(self.beta())
        result = Element()
        for key, c in tensor(self.associator(), self.r_matrix()).terms.items():
            p1, p2, p3, r1, r2 = key
            inner = self.tag(self.antipode(self.elt(p2) * beta * self.antipode(self.elt(p3))))
            term = inner * self.antipode(self.elt(r2)) * alpha * self.elt(r1) * self.elt(p1, c)
            result = result + term

        self._drinfeld_cache = result
        return result

    def gamma(self) -> Element:
        r"""Drinfeld's gamma (V.G. Drinfel'd, "Quasi-Hopf Algebras",
        Leningrad Math. J. 1 (1990), No. 6, eq (1.24)), an element of
        H (x) H used to build ``f_element()``/``f_element_inv()`` below.
        Our Phi is Drinfeld's Phi^-1 in this formula (same substitution
        throughout this method).

        ``xx`` is eq (1.24)'s auxiliary formula for
        T_i (x) U_i (x) V_i (x) W_i:

            xx = (1 (x) Phi) . (id (x) id (x) Delta)(Phi^-1)

        and gamma itself, summed over xx's terms (T_i, U_i, V_i, W_i):

            gamma = sum S(U_i).alpha.V_i (x) S(T_i).alpha.W_i

        Needs only the required interface. Cached, see ``monodromy()``.
        Checked against ``f_element()`` directly by
        ``axioms.check_gamma_definition`` (eq (1.35): gamma ==
        F . Delta(alpha)) -- a genuine cross-check, not circular, since
        gamma is built independently of F here.
        """
        cached = getattr(self, "_gamma_cache", None)
        if cached is not None:
            return cached

        Phi = self.tag(self.associator())
        Phiinv = self.tag(self.associator_inv())
        alpha = self.tag(self.alpha())
        one = self.tag(self.unit())

        # (1 (x) Phi) . (id (x) id (x) Delta)(Phi^-1), an element of
        # H^{(x)4} -- Delta spelled as Δ since Phiinv is tagged.
        xx = tensor(one, Phi) * apply_to_factor(Phiinv, 2, Δ)

        result = Element()
        for key, c in xx.terms.items():
            a1, a2, a3, a4 = key
            left = self.antipode(self.elt(a2)) * alpha * self.elt(a3)
            right = self.antipode(self.elt(a1)) * alpha * self.elt(a4, c)
            result = result + tensor(left, right)

        self._gamma_cache = result
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

        Phi = self.tag(self.associator())
        beta = self.tag(self.beta())
        ga = self.tag(self.gamma())

        result = Element()
        for key, c in Phi.terms.items():
            a1, a2, a3 = key
            part1 = Element()
            for (u, v), cc in Δ(self.elt(a1)).terms.items():
                part1 = part1 + tensor(self.tag(self.antipode(self.elt(v, cc))), self.antipode(self.elt(u)))
            inner = self.elt(a2, c) * beta * self.antipode(self.elt(a3))
            part3 = Δ(inner)
            result = result + part1 * ga * part3

        self._f_element_cache = result
        return result

    def f_element_inv(self) -> Element:
        r"""The inverse of ``f_element()`` (F), per Drinfeld's paper
        (V.G. Drinfel'd, "Quasi-Hopf Algebras", Leningrad Math. J. 1
        (1990), No. 6). Our Phi is Drinfeld's Phi^-1 in the auxiliary
        formulas below (eq (1.24)/(1.25) -- same substitution as in
        ``f_element()``'s ``xx``/``ga``, which this mirrors); Phi = sum
        P (x) Q (x) R itself (eq (1.36)) needs no such substitution.
        Needs only the required interface plus ``comul()`` -- notably
        *not* ``antipode_inv()``, unlike ``qR()``/``pL()``/``U()``/etc.
        Cached, see ``monodromy()``.

        ``yy`` is the F^-1 analogue of ``xx`` -- eq (1.24)'s auxiliary
        formula for K_j (x) L_j (x) M_j (x) N_j, substituting
        Phi_Drinfeld = Phi^-1:

            yy = (Delta (x) id (x) id)(Phi^-1) . (Phi (x) 1)

        ``de`` is the F^-1 analogue of ``ga`` -- eq (1.25)'s delta,
        summed over yy's terms (K_j, L_j, M_j, N_j):

            de = sum K_j.beta.S(N_j) (x) L_j.beta.S(M_j)

        Finally, eq (1.36)'s second line assembles F^-1 itself, summed
        over Phi's own terms (P, Q, R), using Delta' = Delta^op = flip
        o Delta:

            F^-1 = sum Delta(S(P).alpha.Q) . de . (S (x) S)(Delta'(R))

        Verified against ``f_element()`` (F.F^-1 == 1(x)1 == F^-1.F)
        for every example algebra, and against SymplecticFermionQ's own
        independently-derived ``f_element_inv()`` (see its docstring) --
        both now redundant with this generic formula, kept as regression
        tests (``tests/test_symplectic_fermion.py``) rather than
        removed, same policy as ``drinfeld()``/``f_element()``.
        """
        cached = getattr(self, "_f_element_inv_cache", None)
        if cached is not None:
            return cached

        Phi = self.tag(self.associator())
        Phiinv = self.tag(self.associator_inv())
        alpha, beta = self.tag(self.alpha()), self.tag(self.beta())
        one = self.tag(self.unit())

        yy = apply_to_factor(Phiinv, 0, Δ) * tensor(Phi, one)

        de = Element()
        for key, c in yy.terms.items():
            k1, k2, k3, k4 = key
            left = self.elt(k1) * beta * self.antipode(self.elt(k4))
            right = self.elt(k2, c) * beta * self.antipode(self.elt(k3))
            de = de + tensor(left, right)

        result = Element()
        for key, c in Phi.terms.items():
            P, Q, R = key
            left = Δ(self.antipode(self.elt(P)) * alpha * self.elt(Q, c))
            right = self._S_both(flip(Δ(self.elt(R))))
            result = result + left * de * right

        self._f_element_inv_cache = result
        return result

    def qR(self) -> Element:
        r"""q^R = Psi_1 (x) S^-1(alpha.Psi_3).Psi_2, Psi = Phi^-1 -- one
        of four auxiliary elements of H (x) H (Hausser-Nill; eq (3.24)
        in arXiv:1812.10445) used to build ``U()``/``V()``/``Ucop()``/
        ``Vcop()``, needed for ``axioms.check_left_cointegral``/
        ``axioms.check_right_cointegral``. Needs ``antipode_inv()``
        (optional); raises ``NotImplementedError`` if not implemented.
        Cached, see ``monodromy()``.
        """
        cached = getattr(self, "_qR_cache", None)
        if cached is not None:
            return cached
        alpha = self.tag(self.alpha())
        result = Element()
        for key, c in self.associator_inv().terms.items():
            p1, p2, p3 = key
            inner = self.antipode_inv(alpha * self.elt(p3))
            right = inner * self.elt(p2)
            result = result + tensor(self.elt(p1, c), right)
        self._qR_cache = result
        return result

    def pR(self) -> Element:
        r"""p^R = Phi_1 (x) Phi_2.beta.S(Phi_3) -- see ``qR()``. Needs
        only the required interface. Cached, see ``monodromy()``."""
        cached = getattr(self, "_pR_cache", None)
        if cached is not None:
            return cached
        beta = self.tag(self.beta())
        result = Element()
        for key, c in self.associator().terms.items():
            a1, a2, a3 = key
            right = self.elt(a2) * beta * self.antipode(self.elt(a3))
            result = result + tensor(self.elt(a1, c), right)
        self._pR_cache = result
        return result

    def qL(self) -> Element:
        r"""q^L = S(Phi_1).alpha.Phi_2 (x) Phi_3 -- see ``qR()``. Needs
        only the required interface. Cached, see ``monodromy()``."""
        cached = getattr(self, "_qL_cache", None)
        if cached is not None:
            return cached
        alpha = self.tag(self.alpha())
        result = Element()
        for key, c in self.associator().terms.items():
            a1, a2, a3 = key
            left = self.antipode(self.elt(a1)) * alpha * self.elt(a2)
            result = result + tensor(left, self.elt(a3, c))
        self._qL_cache = result
        return result

    def pL(self) -> Element:
        r"""p^L = Psi_2.S^-1(Psi_1.beta) (x) Psi_3, Psi = Phi^-1 -- see
        ``qR()``. Needs ``antipode_inv()`` (optional); raises
        ``NotImplementedError`` if not implemented. Cached, see
        ``monodromy()``."""
        cached = getattr(self, "_pL_cache", None)
        if cached is not None:
            return cached
        beta = self.tag(self.beta())
        result = Element()
        for key, c in self.associator_inv().terms.items():
            p1, p2, p3 = key
            inner = self.antipode_inv(self.elt(p1) * beta)
            left = self.elt(p2) * inner
            result = result + tensor(left, self.elt(p3, c))
        self._pL_cache = result
        return result

    def _S_both(self, elem: Element) -> Element:
        """(S (x) S) applied to a 2-fold tensor element, factor by
        factor. Private helper for U()/V()/Ucop()/Vcop() below."""
        return apply_to_factor(apply_to_factor(elem, 0, self.antipode), 1, self.antipode)

    def _Sinv_both(self, elem: Element) -> Element:
        """(S^-1 (x) S^-1), see ``_S_both``. Needs ``antipode_inv()``
        (optional); raises ``NotImplementedError`` if not implemented."""
        return apply_to_factor(apply_to_factor(elem, 0, self.antipode_inv), 1, self.antipode_inv)

    def U(self) -> Element:
        r"""U = F^-1 . (S (x) S)(q^R_21), an element of H (x) H
        (eq (3.25) in arXiv:1812.10445) -- used by
        ``axioms.check_left_cointegral``. Needs ``f_element_inv()`` and
        ``antipode_inv()`` (both optional); raises
        ``NotImplementedError`` if either isn't implemented. Cached, see
        ``monodromy()``."""
        cached = getattr(self, "_U_cache", None)
        if cached is not None:
            return cached
        Finv = self.tag(self.f_element_inv())
        qR21 = flip(self.tag(self.qR()))
        result = Finv * self._S_both(qR21)
        self._U_cache = result
        return result

    def V(self) -> Element:
        r"""V = (S^-1 (x) S^-1)(F_21 . p^R_21) -- see ``U()``. Needs
        ``antipode_inv()`` (optional); raises ``NotImplementedError`` if
        not implemented. Cached, see ``monodromy()``."""
        cached = getattr(self, "_V_cache", None)
        if cached is not None:
            return cached
        F, pR = self.tag(self.f_element()), self.tag(self.pR())
        result = self._Sinv_both(flip(F * pR))
        self._V_cache = result
        return result

    def Ucop(self) -> Element:
        r"""Ucop = (S^-1 (x) S^-1)(q^L . F^-1) -- see ``U()``. Needs
        ``f_element_inv()`` and ``antipode_inv()`` (both optional);
        raises ``NotImplementedError`` if either isn't implemented.
        Cached, see ``monodromy()``."""
        cached = getattr(self, "_Ucop_cache", None)
        if cached is not None:
            return cached
        qL, Finv = self.tag(self.qL()), self.tag(self.f_element_inv())
        result = self._Sinv_both(qL * Finv)
        self._Ucop_cache = result
        return result

    def Vcop(self) -> Element:
        r"""Vcop = (S (x) S)(p^L) . F_21 -- see ``U()``. Needs
        ``antipode_inv()`` (optional); raises ``NotImplementedError`` if
        not implemented. Cached, see ``monodromy()``."""
        cached = getattr(self, "_Vcop_cache", None)
        if cached is not None:
            return cached
        pL, F = self.tag(self.pL()), self.tag(self.f_element())
        result = self._S_both(pL) * flip(F)
        self._Vcop_cache = result
        return result
