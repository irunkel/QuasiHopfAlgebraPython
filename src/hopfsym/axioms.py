"""Generic, algebra-agnostic checks for the quasi-Hopf algebra axioms.

Every function here takes a :class:`~hopfsym.algebra.QuasiHopfAlgebra`
instance and some basis elements to test against, and returns True/False
(printing a description of the first failure it finds, if any). They are
written purely against the QuasiHopfAlgebra interface, so they work
unchanged for any algebra that implements it -- these mirror the
``test...`` routines in the original Mathematica code (testAssoc,
testCoassoc, testPentagon, testHopf, testAntipode, testEvalCoeval,
testRinter, testHexagon, testRibbon, testSDeltaR).

For an honest Hopf algebra (Phi = 1 (x) 1 (x) 1, alpha = beta = 1),
``check_twisted_coassociativity`` reduces to ordinary coassociativity
and ``check_antipode`` reduces to the usual antipode axiom -- so nothing
here is specific to the quasi-Hopf case, it is simply general enough to
cover it.

``check_r_matrix_intertwiner``, ``check_hexagon``, ``check_ribbon`` and
``check_s_delta_compatibility`` need the *optional* part of the
interface (``r_matrix``/``ribbon``/``drinfeld``/``f_element`` -- see
``algebra.py``), for algebras that are additionally quasi-triangular or
ribbon. They are not part of ``check_all`` since most algebras won't
implement that optional structure; call them separately when they do.

By default, checks that take a single basis element (counit, twisted
coassociativity, antipode) test a random sample rather than the whole
basis, same as the original Mathematica code's ``ntest``-many random
tests -- for algebras with more than a handful of basis elements,
exhaustive per-element checking gets expensive fast (each one expands
the coproduct, which itself can have many terms). Pass
``samples=alg.basis()`` explicitly for an exhaustive check when you
want one (e.g. in a one-off regression test for a small algebra).

The sample itself is drawn with a *fresh random seed every call* (not a
fixed one) -- so running the suite twice exercises different basis
elements each time, rather than silently re-checking the same handful
forever. The tradeoff (a failure isn't automatically reproducible just
by re-running) is covered by every check printing the seed it used the
moment it finds a failure -- unconditionally, not gated by ``verbose``,
since that's the one piece of failure output you always want, even in
a quiet test run. ``SAMPLE_SIZE`` (module-level, not baked into any
function's default arguments, so it can be changed at runtime, e.g.
``axioms.SAMPLE_SIZE = 20`` for a slower but more thorough run) is the
shared baseline sample count; ``check_r_matrix_intertwiner`` is the one
check expensive enough (exact arithmetic against a several-hundred-term
R-matrix) to default to a fraction of it instead.

Basis elements are built as *tagged* Elements (``alg.elt(key)``, not the
untagged ``Element.basis(key)``), and structure maps are spelled with
their tag-dispatching wrappers (``Δ`` for comul, ``ε`` for counit) and
``*``/``**`` for the algebra product -- so e.g. associativity reads as
``(ea * eb) * ec == ea * (eb * ec)`` rather than
``alg.mul(alg.mul(ea, eb), ec) == alg.mul(ea, alg.mul(eb, ec))``, and a
product of two same-arity tensor elements (like ``Δ(a) * Δ(b)`` in
H (x) H) uses ``*``'s arity-aware dispatch to ``tensor_mul`` instead of
spelling that out explicitly (see ``element.py``'s docstrings for both).
The one-off "nullary" structure elements (``alg.associator()``,
``alg.r_matrix()``, ...) come back untagged from the algebra itself
(their own implementations build them from untagged pieces internally);
``alg.tag(...)`` gives a tagged copy where a check actually needs one
-- either because ``Δ``/``ε`` need a tagged argument directly (no
other operand around to supply the tag via ``*``'s wildcard), or
because *both* sides of a product would otherwise be untagged. Where
neither applies, an untagged structure element is left as-is: ``*``'s
``None``-is-a-wildcard convention (``_combine_alg``) lets it combine
fine with a tagged operand on the other side.
"""

from __future__ import annotations

import random

from .element import Element, TensorKey, apply_to_factor, flip, permute_factors, tensor, Δ, ε

# Baseline number of random basis elements/pairs/triples a check samples
# when the caller doesn't pass `samples=` explicitly. Deliberately looked
# up at call time (see _sample_* below) rather than baked into a function
# default, so `axioms.SAMPLE_SIZE = ...` actually takes effect.
SAMPLE_SIZE = 6


def _new_seed() -> int:
    """A fresh random seed from OS entropy, used whenever a _sample_*
    call isn't given an explicit one."""
    return random.SystemRandom().randrange(2**31)


def _report_seed(seed) -> None:
    """Print the seed a failing check's random sample used, so the
    failure can be reproduced -- unconditional (not gated by verbose):
    this is the one thing worth seeing even in an otherwise-quiet run."""
    if seed is not None:
        print(f"  (random sample seed: {seed})")


def _sample_pairs(alg, n=None, seed=None):
    if n is None:
        n = SAMPLE_SIZE
    if seed is None:
        seed = _new_seed()
    rng = random.Random(seed)
    basis = list(alg.basis())
    return [(rng.choice(basis), rng.choice(basis)) for _ in range(n)], seed


def _sample_triples(alg, n=None, seed=None):
    if n is None:
        n = SAMPLE_SIZE
    if seed is None:
        seed = _new_seed()
    rng = random.Random(seed)
    basis = list(alg.basis())
    return [(rng.choice(basis), rng.choice(basis), rng.choice(basis)) for _ in range(n)], seed


def _sample_single(alg, n=None, seed=None):
    if n is None:
        n = SAMPLE_SIZE
    if seed is None:
        seed = _new_seed()
    rng = random.Random(seed)
    basis = list(alg.basis())
    return [rng.choice(basis) for _ in range(min(n, len(basis)))], seed


def _pair(key):
    """Split a 2-fold TensorKey (or bare key, for arity 1) into its two
    factors."""
    factors = key if isinstance(key, TensorKey) else (key,)
    if len(factors) != 2:
        raise ValueError(f"expected a 2-fold tensor key, got {key!r}")
    return factors


def _triple(key):
    """Split a 3-fold TensorKey into its three factors."""
    factors = key if isinstance(key, TensorKey) else (key,)
    if len(factors) != 3:
        raise ValueError(f"expected a 3-fold tensor key, got {key!r}")
    return factors


def check_associativity(alg, samples=None, verbose=True) -> bool:
    """(a*b)*c == a*(b*c) for sample triples of basis elements."""
    seed = None
    if samples is None:
        samples, seed = _sample_triples(alg)
    for a, b, c in samples:
        ea, eb, ec = alg.elt(a), alg.elt(b), alg.elt(c)
        lhs = (ea * eb) * ec
        rhs = ea * (eb * ec)
        if lhs != rhs:
            if verbose:
                print(f"associativity failed for a={a}, b={b}, c={c}")
                print(f"  (ab)c = {lhs}")
                print(f"  a(bc) = {rhs}")
            _report_seed(seed)
            return False
    return True


def check_bialgebra_homomorphism(alg, samples=None, verbose=True) -> bool:
    """Delta(a*b) == Delta(a) * Delta(b) (product taken in H (x) H)."""
    seed = None
    if samples is None:
        samples, seed = _sample_pairs(alg)
    for a, b in samples:
        ea, eb = alg.elt(a), alg.elt(b)
        lhs = Δ(ea * eb)
        rhs = Δ(ea) * Δ(eb)
        if lhs != rhs:
            if verbose:
                print(f"Delta homomorphism property failed for a={a}, b={b}")
                print(f"  Delta(ab)         = {lhs}")
                print(f"  Delta(a) Delta(b) = {rhs}")
            _report_seed(seed)
            return False
    return True


def check_counit(alg, samples=None, verbose=True) -> bool:
    """(id (x) eps)(Delta a) == a  and  (eps (x) id)(Delta a) == a."""
    seed = None
    if samples is None:
        samples, seed = _sample_single(alg)
    for a in samples:
        ea = alg.elt(a)
        da = Δ(ea)
        left = Element()
        right = Element()
        for key, c in da.terms.items():
            k1, k2 = _pair(key)
            left = left + (c * ε(alg.elt(k2))) * alg.elt(k1)
            right = right + (c * ε(alg.elt(k1))) * alg.elt(k2)
        if left != ea or right != ea:
            if verbose:
                print(f"counit axiom failed for a={a}")
                print(f"  (id x eps) Delta(a) = {left}")
                print(f"  (eps x id) Delta(a) = {right}")
                print(f"  a                   = {ea}")
            _report_seed(seed)
            return False
    return True


def check_twisted_coassociativity(alg, samples=None, verbose=True) -> bool:
    """(Delta (x) id)(Delta(a)) * Phi == Phi * (id (x) Delta)(Delta(a)).

    For an honest Hopf algebra (Phi = 1^{(x)3}) this is ordinary
    coassociativity.
    """
    seed = None
    if samples is None:
        samples, seed = _sample_single(alg)
    Phi = alg.associator()
    for a in samples:
        da = Δ(alg.elt(a))
        lhs = apply_to_factor(da, 0, Δ) * Phi
        rhs = Phi * apply_to_factor(da, 1, Δ)
        if lhs != rhs:
            if verbose:
                print(f"twisted coassociativity failed for a={a}")
                print(f"  (Delta x id)(Delta a) . Phi = {lhs}")
                print(f"  Phi . (id x Delta)(Delta a) = {rhs}")
            _report_seed(seed)
            return False
    return True


def check_pentagon(alg, verbose=True) -> bool:
    """The pentagon axiom for the associator:

        (Delta x id x id)(Phi) . (id x id x Delta)(Phi)
            == (Phi (x) 1) . (id x Delta x id)(Phi) . (1 (x) Phi)

    Phi is tagged here (``alg.tag``) since ``apply_to_factor(Phi, ..., Δ)``
    needs its argument tagged for ``Δ`` to dispatch -- there is no other
    operand around at that point to supply the tag via ``*``'s wildcard,
    unlike in most of the other checks below.
    """
    Phi = alg.tag(alg.associator())
    one = alg.unit()

    lhs = apply_to_factor(Phi, 0, Δ) * apply_to_factor(Phi, 2, Δ)
    rhs = tensor(Phi, one) * (apply_to_factor(Phi, 1, Δ) * tensor(one, Phi))

    if lhs != rhs:
        if verbose:
            print("pentagon axiom failed")
            print(f"  lhs = {lhs}")
            print(f"  rhs = {rhs}")
        return False
    return True


def check_r_matrix_intertwiner(alg, samples=None, verbose=True) -> bool:
    """The defining property of a universal R-matrix R (an element of
    H (x) H, from ``alg.r_matrix()``):

        R . Delta(a) == flip(Delta(a)) . R   for every a in H,

    plus its normalisation (id (x) eps)(R) == 1 == (eps (x) id)(R).
    Requires ``alg.r_matrix()`` (optional in the QuasiHopfAlgebra
    interface -- only algebras that are quasi-triangular implement it).

    Defaults to a third of ``SAMPLE_SIZE``: R typically has hundreds of
    terms, and each sample multiplies it out in full exact arithmetic
    twice (``R.Delta(a)`` and ``flip(Delta(a)).R``), so this is the one
    check expensive enough to need a smaller default -- pass
    ``samples=`` explicitly to override.
    """
    seed = None
    if samples is None:
        samples, seed = _sample_single(alg, n=max(1, SAMPLE_SIZE // 3))
    R = alg.r_matrix()
    one = alg.unit()

    left = Element()
    right = Element()
    for key, c in R.terms.items():
        k1, k2 = _pair(key)
        left = left + (c * ε(alg.elt(k2))) * alg.elt(k1)
        right = right + (c * ε(alg.elt(k1))) * alg.elt(k2)
    if left != one or right != one:
        if verbose:
            print("R-matrix intertwiner failed: normalisation")
            print(f"  (id x eps)(R) = {left},  (eps x id)(R) = {right},  1 = {one}")
        return False

    for a in samples:
        da = Δ(alg.elt(a))
        lhs = R * da
        rhs = flip(da) * R
        if lhs != rhs:
            if verbose:
                print(f"R-matrix intertwiner failed for a={a}")
                print(f"  R . Delta(a)       = {lhs}")
                print(f"  flip(Delta(a)) . R = {rhs}")
            _report_seed(seed)
            return False
    return True


def check_hexagon(alg, verbose=True) -> bool:
    r"""The two hexagon axioms tying the associator Phi to the R-matrix R
    together (quasi-triangularity compatible with non-trivial
    associativity):

        Phi_{2,3,1} . (Delta (x) id)(R) . Phi
            == R_{13} . Phi_{1,3,2} . R_{23}

        Phi^-1_{3,1,2} . (id (x) Delta)(R) . Phi^-1
            == R_{13} . Phi^-1_{2,1,3} . R_{12}

    where Phi_{ijk} denotes Phi with its legs permuted (via
    ``permute_factors``) and R_{ij} denotes R padded with a unit leg to
    make it a 3-fold tensor (via ``apply_to_factor`` + ``tensor``).
    Requires ``alg.r_matrix()`` (optional in the interface).

    R is tagged (``alg.tag``) since ``apply_to_factor(R, ..., Δ)`` needs
    its argument tagged for ``Δ`` to dispatch; Phi/Phi^-1 don't need it
    -- every product they appear in also has an R-derived (hence
    tagged) operand, which supplies the algebra via ``*``'s wildcard.
    """
    Phi = alg.associator()
    Phiinv = alg.associator_inv()
    R = alg.tag(alg.r_matrix())
    one = alg.unit()

    def pad_left(v):
        return tensor(one, v)

    def pad_right(u):
        return tensor(u, one)

    phi231 = permute_factors(Phi, (1, 2, 0))
    phi132 = permute_factors(Phi, (0, 2, 1))
    r13 = apply_to_factor(R, 1, pad_left)
    r23 = apply_to_factor(R, 0, pad_left)

    delta_id_R = apply_to_factor(R, 0, Δ)
    lhs1 = phi231 * delta_id_R * Phi
    rhs1 = r13 * phi132 * r23

    if lhs1 != rhs1:
        if verbose:
            print("hexagon axiom (1) failed")
            print(f"  lhs = {lhs1}")
            print(f"  rhs = {rhs1}")
        return False

    phi312inv = permute_factors(Phiinv, (2, 0, 1))
    phi213inv = permute_factors(Phiinv, (1, 0, 2))
    r12 = apply_to_factor(R, 1, pad_right)

    id_delta_R = apply_to_factor(R, 1, Δ)
    lhs2 = phi312inv * id_delta_R * Phiinv
    rhs2 = r13 * phi213inv * r12

    if lhs2 != rhs2:
        if verbose:
            print("hexagon axiom (2) failed")
            print(f"  lhs = {lhs2}")
            print(f"  rhs = {rhs2}")
        return False

    return True


def check_antipode(alg, samples=None, verbose=True) -> bool:
    """For Delta(a) = sum a_1 (x) a_2:

        sum S(a_1) . alpha . a_2 == eps(a) alpha
        sum a_1 . beta . S(a_2)  == eps(a) beta

    For an honest Hopf algebra (alpha = beta = 1) this is the usual
    antipode axiom.

    alpha/beta are tagged (``alg.tag``) since ``S(a_1)`` (``alg.antipode``'s
    result) always comes back untagged -- with neither side of
    ``S(a_1) * alpha`` otherwise tagged, ``*`` would have no algebra to
    resolve to.
    """
    seed = None
    if samples is None:
        samples, seed = _sample_single(alg)
    alpha, beta = alg.tag(alg.alpha()), alg.tag(alg.beta())
    for a in samples:
        ea = alg.elt(a)
        da = Δ(ea)
        eps_a = ε(ea)

        eq1 = Element()
        eq2 = Element()
        for key, c in da.terms.items():
            k1, k2 = _pair(key)
            a1c, a2 = alg.elt(k1, c), alg.elt(k2)
            a1, a2c = alg.elt(k1), alg.elt(k2, c)
            eq1 = eq1 + alg.antipode(a1c) * alpha * a2
            eq2 = eq2 + a1 * beta * alg.antipode(a2c)

        target1 = eps_a * alpha
        target2 = eps_a * beta
        if eq1 != target1 or eq2 != target2:
            if verbose:
                print(f"antipode axiom failed for a={a}")
                print(f"  sum S(a_1) alpha a_2 = {eq1},  eps(a) alpha = {target1}")
                print(f"  sum a_1 beta S(a_2)  = {eq2},  eps(a) beta  = {target2}")
            _report_seed(seed)
            return False
    return True


def check_evaluation_coevaluation(alg, verbose=True) -> bool:
    r"""The evaluation/coevaluation identities for the associator,
    antipode and alpha/beta -- part of the rigidity data of a quasi-Hopf
    algebra, alongside ``check_antipode`` (mirrors ``testEvalCoeval`` in
    the original Mathematica code):

        sum S(a) . alpha . b . beta . S(c) == 1   (Phi = sum a (x) b (x) c)
        sum a . beta . S(b) . alpha . c    == 1   (Phi^-1 = sum a (x) b (x) c)

    Uses only the required part of the QuasiHopfAlgebra interface.
    Phi/Phi^-1 stay untagged (only iterated over via ``.terms.items()``,
    never multiplied directly); alpha/beta are tagged for the same
    reason as in ``check_antipode``.
    """
    Phi = alg.associator()
    Phiinv = alg.associator_inv()
    alpha, beta = alg.tag(alg.alpha()), alg.tag(alg.beta())
    one = alg.unit()

    t1 = Element()
    for key, c in Phi.terms.items():
        a, b, cc = _triple(key)
        term = alg.antipode(alg.elt(a)) * alpha * alg.elt(b) * beta * alg.antipode(alg.elt(cc))
        for k2, c2 in term.terms.items():
            t1.add_term(k2, c * c2)
    if t1 != one:
        if verbose:
            print(f"evaluation identity failed: sum S(a) alpha b beta S(c) = {t1}, expected {one}")
        return False

    t2 = Element()
    for key, c in Phiinv.terms.items():
        a, b, cc = _triple(key)
        term = alg.elt(a) * beta * alg.antipode(alg.elt(b)) * alpha * alg.elt(cc)
        for k2, c2 in term.terms.items():
            t2.add_term(k2, c * c2)
    if t2 != one:
        if verbose:
            print(f"coevaluation identity failed: sum a beta S(b) alpha c = {t2}, expected {one}")
        return False

    return True


def check_ribbon(alg, verbose=True) -> bool:
    r"""Properties defining a ribbon element v = ``alg.ribbon()``, given a
    quasi-triangular structure R = ``alg.r_matrix()`` and Drinfeld
    element u = ``alg.drinfeld()`` (all optional in the interface;
    mirrors ``testRibbon`` in the original Mathematica code):

        S(v) == v
        eps(v) == 1
        u . S(u) == v . v
        M . Delta(v) == v (x) v      (M = R_21 . R, the monodromy)

    v/u/R are each tagged (``alg.tag``): ``ε(v)`` needs v tagged
    directly, and ``u . S(u)``/``M = flip(R) . R`` each have *no* tagged
    operand otherwise (``alg.antipode(u)``/``flip(R)`` of an untagged
    argument stay untagged too), unlike most of the other checks where
    one side of a product is already tagged.
    """
    v = alg.tag(alg.ribbon())
    one = alg.unit()

    if alg.antipode(v) != v:
        if verbose:
            print(f"ribbon axiom failed: S(v) != v  (S(v) = {alg.antipode(v)}, v = {v})")
        return False

    if ε(v) != 1:
        if verbose:
            print(f"ribbon axiom failed: eps(v) != 1  (eps(v) = {ε(v)})")
        return False

    u = alg.tag(alg.drinfeld())
    lhs = u * alg.antipode(u)
    rhs = v * v
    if lhs != rhs:
        if verbose:
            print(f"ribbon axiom failed: u.S(u) != v.v  (u.S(u) = {lhs}, v.v = {rhs})")
        return False

    R = alg.tag(alg.r_matrix())
    M = flip(R) * R
    lhs = M * Δ(v)
    rhs = tensor(v, v)
    if lhs != rhs:
        if verbose:
            print(f"ribbon axiom failed: M.Delta(v) != v (x) v  (M.Delta(v) = {lhs}, v (x) v = {rhs})")
        return False

    return True


def check_s_delta_compatibility(alg, samples=None, verbose=True) -> bool:
    r"""Compatibility of the coproduct and antipode via the element
    F = ``alg.f_element()`` (Drinfeld's paper; mirrors ``testSDeltaR`` in
    the original Mathematica code):

        (S (x) S)(flip(Delta(a))) . F == F . Delta(S(a))   for every a.

    Requires ``alg.f_element()`` (optional in the interface). F is
    tagged (``alg.tag``) since the hand-built ``swapped_S`` (accumulated
    term-by-term below, not via ``tensor()``/``*``) stays untagged --
    without it, neither side of ``swapped_S . F`` would be tagged.
    ``F . Delta(S(a))`` uses ``alg.comul`` directly rather than ``Δ``:
    ``alg.antipode(...)``'s result is untagged, and unlike ``*`` (which
    accepts one untagged operand via its wildcard), ``Δ`` needs its
    single argument tagged with nothing else around to supply it.
    """
    seed = None
    if samples is None:
        samples, seed = _sample_single(alg)
    F = alg.tag(alg.f_element())

    for a in samples:
        ea = alg.elt(a)
        da = Δ(ea)

        # (S (x) S) applied to flip(Delta(a)):
        swapped_S = Element()
        for key, c in flip(da).terms.items():
            u, v = _pair(key)
            piece = tensor(alg.antipode(alg.elt(u)), alg.antipode(alg.elt(v)))
            for k2, c2 in piece.terms.items():
                swapped_S.add_term(k2, c * c2)

        lhs = swapped_S * F
        rhs = F * alg.comul(alg.antipode(ea))
        if lhs != rhs:
            if verbose:
                print(f"S-Delta compatibility failed for a={a}")
                print(f"  (S x S)(flip(Delta(a))) . F = {lhs}")
                print(f"  F . Delta(S(a))             = {rhs}")
            _report_seed(seed)
            return False
    return True


def check_all(alg, verbose=True) -> bool:
    """Run every check above; returns True iff all of them pass."""
    checks = [
        ("associativity", lambda: check_associativity(alg, verbose=verbose)),
        ("bialgebra homomorphism", lambda: check_bialgebra_homomorphism(alg, verbose=verbose)),
        ("counit", lambda: check_counit(alg, verbose=verbose)),
        ("twisted coassociativity", lambda: check_twisted_coassociativity(alg, verbose=verbose)),
        ("pentagon", lambda: check_pentagon(alg, verbose=verbose)),
        ("antipode", lambda: check_antipode(alg, verbose=verbose)),
        ("evaluation/coevaluation", lambda: check_evaluation_coevaluation(alg, verbose=verbose)),
    ]
    ok = True
    for name, fn in checks:
        passed = fn()
        if verbose:
            print(f"{'PASS' if passed else 'FAIL'}: {name}")
        ok = ok and passed
    return ok
