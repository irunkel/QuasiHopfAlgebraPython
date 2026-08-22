"""Independent closed-form formulas for drinfeld()/f_element()/
f_element_inv(), used to regression-test QuasiHopfAlgebra's generic
implementations of those (see algebra.py) against something derived
independently of that generic code path -- for the two examples where
such a closed form is available (RestrictedSl2: trivial associator
collapses the general formula to the classical Hopf-algebra one;
SymplecticFermionQ: the paper's own "Some special elements of Q" section,
eq:def:F-Q and eq:sqs+sqsinvbr-Q, plus an elementary block-by-block
inversion of eq:def:F-Q for f_element_inv() -- not itself sourced from
the paper, see symplectic_fermion_expected_f_element_inv's docstring).

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


def symplectic_fermion_expected_f_element_inv(alg) -> Element:
    """F^-1 = e0 (x) 1 + e1 (x) (K^-N.e0) + (beta^-2(-iK)^-N.e1) (x) e1 --
    an elementary block-by-block inversion of eq:def:F-Q (see
    symplectic_fermion_expected_f_element above): H (x) H is the
    tensor-product *algebra* (componentwise multiplication), and
    e0 (x) e0, e0 (x) e1, e1 (x) e0, e1 (x) e1 are orthogonal idempotents
    summing to 1 (x) 1 that F splits into (term 1 spans the first two,
    terms 2 and 3 are each already within a single block), each
    invertible block-by-block -- not itself a formula from the paper, but
    confirmed (via QuasiHopfAlgebra.f_element_inv(), the generic
    implementation of Drinfeld's eq (1.36)) to be the actual inverse of
    the paper's own F, and used here as an independent construction to
    cross-check that generic implementation against."""
    one, e0, e1 = alg.unit(), alg.e0, alg.e1
    KN_inv = alg._K(-alg.N)
    scalar3_inv = (alg.beta_scalar() ** 2 * (-alg.i()) ** alg.N).inverse()
    term1 = tensor(e0, one)
    term2 = tensor(e1, alg.mul(KN_inv, e0))
    term3 = tensor(alg.mul(scalar3_inv * KN_inv, e1), e1)
    return term1 + term2 + term3


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
