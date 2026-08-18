"""Generic, algebra-agnostic checks for the quasi-Hopf algebra axioms.

Every function here takes a :class:`~hopfsym.algebra.QuasiHopfAlgebra`
instance and some basis elements to test against, and returns True/False
(printing a description of the first failure it finds, if any). They are
written purely against the QuasiHopfAlgebra interface, so they work
unchanged for any algebra that implements it -- these mirror the
``test...`` routines in the original Mathematica code (testAssoc,
testCoassoc, testPentagon, testHopf, testAntipode).

For an honest Hopf algebra (Phi = 1 (x) 1 (x) 1, alpha = beta = 1),
``check_twisted_coassociativity`` reduces to ordinary coassociativity
and ``check_antipode`` reduces to the usual antipode axiom -- so nothing
here is specific to the quasi-Hopf case, it is simply general enough to
cover it.

By default, checks that take a single basis element (counit, twisted
coassociativity, antipode) test a random sample rather than the whole
basis, same as the original Mathematica code's ``ntest``-many random
tests -- for algebras with more than a handful of basis elements,
exhaustive per-element checking gets expensive fast (each one expands
the coproduct, which itself can have many terms). Pass
``samples=alg.basis()`` explicitly for an exhaustive check when you
want one (e.g. in a one-off regression test for a small algebra).
"""

from __future__ import annotations

import random

from .element import Element, TensorKey, apply_to_factor, tensor_mul


def _sample_pairs(alg, n=6, seed=0):
    rng = random.Random(seed)
    basis = list(alg.basis())
    return [(rng.choice(basis), rng.choice(basis)) for _ in range(n)]


def _sample_triples(alg, n=6, seed=0):
    rng = random.Random(seed)
    basis = list(alg.basis())
    return [(rng.choice(basis), rng.choice(basis), rng.choice(basis)) for _ in range(n)]


def _sample_single(alg, n=10, seed=0):
    rng = random.Random(seed)
    basis = list(alg.basis())
    return [rng.choice(basis) for _ in range(min(n, len(basis)))]


def _pair(key):
    """Split a 2-fold TensorKey (or bare key, for arity 1) into its two
    factors."""
    factors = key if isinstance(key, TensorKey) else (key,)
    if len(factors) != 2:
        raise ValueError(f"expected a 2-fold tensor key, got {key!r}")
    return factors


def check_associativity(alg, samples=None, verbose=True) -> bool:
    """(a*b)*c == a*(b*c) for sample triples of basis elements."""
    samples = samples if samples is not None else _sample_triples(alg)
    for a, b, c in samples:
        ea, eb, ec = Element.basis(a), Element.basis(b), Element.basis(c)
        lhs = alg.mul(alg.mul(ea, eb), ec)
        rhs = alg.mul(ea, alg.mul(eb, ec))
        if lhs != rhs:
            if verbose:
                print(f"associativity failed for a={a}, b={b}, c={c}")
                print(f"  (ab)c = {lhs}")
                print(f"  a(bc) = {rhs}")
            return False
    return True


def check_bialgebra_homomorphism(alg, samples=None, verbose=True) -> bool:
    """Delta(a*b) == Delta(a) * Delta(b) (product taken in H (x) H)."""
    samples = samples if samples is not None else _sample_pairs(alg)
    for a, b in samples:
        ea, eb = Element.basis(a), Element.basis(b)
        lhs = alg.comul(alg.mul(ea, eb))
        rhs = tensor_mul(alg, alg.comul(ea), alg.comul(eb))
        if lhs != rhs:
            if verbose:
                print(f"Delta homomorphism property failed for a={a}, b={b}")
                print(f"  Delta(ab)         = {lhs}")
                print(f"  Delta(a) Delta(b) = {rhs}")
            return False
    return True


def check_counit(alg, samples=None, verbose=True) -> bool:
    """(id (x) eps)(Delta a) == a  and  (eps (x) id)(Delta a) == a."""
    samples = samples if samples is not None else _sample_single(alg)
    for a in samples:
        ea = Element.basis(a)
        da = alg.comul(ea)
        left = Element()
        right = Element()
        for key, c in da.terms.items():
            k1, k2 = _pair(key)
            left = left + (c * alg.counit(Element.basis(k2))) * Element.basis(k1)
            right = right + (c * alg.counit(Element.basis(k1))) * Element.basis(k2)
        if left != ea or right != ea:
            if verbose:
                print(f"counit axiom failed for a={a}")
                print(f"  (id x eps) Delta(a) = {left}")
                print(f"  (eps x id) Delta(a) = {right}")
                print(f"  a                   = {ea}")
            return False
    return True


def check_twisted_coassociativity(alg, samples=None, verbose=True) -> bool:
    """(Delta (x) id)(Delta(a)) * Phi == Phi * (id (x) Delta)(Delta(a)).

    For an honest Hopf algebra (Phi = 1^{(x)3}) this is ordinary
    coassociativity.
    """
    samples = samples if samples is not None else _sample_single(alg)
    Phi = alg.associator()
    for a in samples:
        da = alg.comul(Element.basis(a))
        lhs = tensor_mul(alg, apply_to_factor(da, 0, alg.comul), Phi)
        rhs = tensor_mul(alg, Phi, apply_to_factor(da, 1, alg.comul))
        if lhs != rhs:
            if verbose:
                print(f"twisted coassociativity failed for a={a}")
                print(f"  (Delta x id)(Delta a) . Phi = {lhs}")
                print(f"  Phi . (id x Delta)(Delta a) = {rhs}")
            return False
    return True


def check_pentagon(alg, verbose=True) -> bool:
    """The pentagon axiom for the associator:

        (Delta x id x id)(Phi) . (id x id x Delta)(Phi)
            == (Phi (x) 1) . (id x Delta x id)(Phi) . (1 (x) Phi)
    """
    from .element import tensor

    Phi = alg.associator()
    one = alg.unit()

    lhs = tensor_mul(alg, apply_to_factor(Phi, 0, alg.comul), apply_to_factor(Phi, 2, alg.comul))

    rhs = tensor_mul(
        alg,
        tensor(Phi, one),
        tensor_mul(alg, apply_to_factor(Phi, 1, alg.comul), tensor(one, Phi)),
    )

    if lhs != rhs:
        if verbose:
            print("pentagon axiom failed")
            print(f"  lhs = {lhs}")
            print(f"  rhs = {rhs}")
        return False
    return True


def check_antipode(alg, samples=None, verbose=True) -> bool:
    """For Delta(a) = sum a_1 (x) a_2:

        sum S(a_1) . alpha . a_2 == eps(a) alpha
        sum a_1 . beta . S(a_2)  == eps(a) beta

    For an honest Hopf algebra (alpha = beta = 1) this is the usual
    antipode axiom.
    """
    samples = samples if samples is not None else _sample_single(alg)
    alpha, beta = alg.alpha(), alg.beta()
    for a in samples:
        ea = Element.basis(a)
        da = alg.comul(ea)
        eps_a = alg.counit(ea)

        eq1 = Element()
        eq2 = Element()
        for key, c in da.terms.items():
            k1, k2 = _pair(key)
            a1c, a2 = Element.basis(k1, c), Element.basis(k2)
            a1, a2c = Element.basis(k1), Element.basis(k2, c)
            eq1 = eq1 + alg.mul(alg.mul(alg.antipode(a1c), alpha), a2)
            eq2 = eq2 + alg.mul(alg.mul(a1, beta), alg.antipode(a2c))

        target1 = eps_a * alpha
        target2 = eps_a * beta
        if eq1 != target1 or eq2 != target2:
            if verbose:
                print(f"antipode axiom failed for a={a}")
                print(f"  sum S(a_1) alpha a_2 = {eq1},  eps(a) alpha = {target1}")
                print(f"  sum a_1 beta S(a_2)  = {eq2},  eps(a) beta  = {target2}")
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
    ]
    ok = True
    for name, fn in checks:
        passed = fn()
        if verbose:
            print(f"{'PASS' if passed else 'FAIL'}: {name}")
        ok = ok and passed
    return ok
