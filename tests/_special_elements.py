"""Independent closed-form formulas for drinfeld()/f_element(), used to
regression-test QuasiHopfAlgebra's generic implementations of those two
(see algebra.py) against something derived independently of that
generic code path -- for the two examples where such a closed form is
available (RestrictedSl2: trivial associator collapses the general
formula to the classical Hopf-algebra one; SymplecticFermionQ: the
paper's own "Some special elements of Q" section, eq:def:F-Q and
eq:sqs+sqsinvbr-Q).

Not a test module itself (no test_ prefix, so unittest discovery skips
it); imported by the test_*.py files that assert equality against these,
and by soak_test.py, which runs the same comparisons as an additional
per-round check -- one source of truth for both, rather than duplicating
the formulas in each place.
"""

from __future__ import annotations

from hopfsym.element import Element, tensor


def restricted_sl2_expected_drinfeld(alg) -> Element:
    """u = sum S(r2) . r1, for R = sum r1 (x) r2 -- the classical
    Hopf-algebra Drinfeld element formula, valid here because
    RestrictedSl2's associator is (unconditionally) trivial."""
    expected = Element()
    for key, c in alg.r_matrix().terms.items():
        r1, r2 = key
        term = alg.mul(alg.antipode(Element.basis(r2)), Element.basis(r1))
        for k2, c2 in term.terms.items():
            expected.add_term(k2, c * c2)
    return expected


def restricted_sl2_expected_f_element(alg) -> Element:
    """F = 1 (x) 1 -- trivial, since RestrictedSl2's associator is
    (unconditionally) trivial."""
    one = alg.unit()
    return tensor(one, one)


def symplectic_fermion_expected_f_element(alg) -> Element:
    """F = e0 (x) 1 + e1 (x) (K^N . e0) + beta^2(-iK)^N.e1 (x) e1
    (eq:def:F-Q in the paper, Section 3.2) -- stated there as a general
    fact for every (N, beta), not a special case."""
    one, e0, e1 = alg.unit(), alg.e0, alg.e1
    KN = alg._K(alg.N)
    scalar3 = alg.beta_scalar() ** 2 * (-alg.i()) ** alg.N
    return tensor(e0, one) + tensor(e1, alg.mul(KN, e0)) + tensor(alg.mul(scalar3 * KN, e1), e1)


def symplectic_fermion_expected_drinfeld(alg) -> Element:
    """u = (e0.K + e1.beta.(-iK)^N) . prod_{i=1}^N (1 - 2 f_i^+ f_i^-)
    (eq:sqs+sqsinvbr-Q in the paper, Section 3.2) -- likewise general,
    not a special case."""
    one, e0, e1 = alg.unit(), alg.e0, alg.e1
    K1, KN = alg._K(1), alg._K(alg.N)
    scalar_u = alg.beta_scalar() * (-alg.i()) ** alg.N
    prefactor = alg.mul(e0, K1) + alg.mul(e1, scalar_u * KN)
    product = one
    for i in range(alg.N):
        fi_plus, fi_minus = alg._f(2 * i), alg._f(2 * i + 1)
        product = alg.mul(product, one - 2 * alg.mul(fi_plus, fi_minus))
    return alg.mul(prefactor, product)
