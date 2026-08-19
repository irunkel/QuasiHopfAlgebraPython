"""Regression tests for U_res sl(2) (K^p = 1), checking it against every
required quasi-Hopf axiom in hopfsym.axioms -- see restricted_sl2.py's
module docstring for the precise definition and how it relates to
QuantumSl2Quasi. It is an honest Hopf algebra (trivial associator), so
check_all should pass exactly the same way it does for that reason
alone (Phi = 1(x)1(x)1, alpha = beta = 1).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hopfsym import axioms
from hopfsym.element import Element, TensorKey
from hopfsym.examples import QuantumSl2Quasi, RestrictedSl2

from _random_model import random_p, random_p_t


class RestrictedSl2AxiomTests(unittest.TestCase):
    def _check(self, p):
        alg = RestrictedSl2(p=p)
        with self.subTest(p=p, axiom="associativity"):
            self.assertTrue(axioms.check_associativity(alg, verbose=True))
        with self.subTest(p=p, axiom="bialgebra_homomorphism"):
            self.assertTrue(axioms.check_bialgebra_homomorphism(alg, verbose=True))
        with self.subTest(p=p, axiom="counit"):
            self.assertTrue(axioms.check_counit(alg, verbose=True))
        with self.subTest(p=p, axiom="twisted_coassociativity"):
            # Phi is trivial, so this is ordinary coassociativity.
            self.assertTrue(axioms.check_twisted_coassociativity(alg, verbose=True))
        with self.subTest(p=p, axiom="pentagon"):
            self.assertTrue(axioms.check_pentagon(alg, verbose=True))
        with self.subTest(p=p, axiom="antipode"):
            self.assertTrue(axioms.check_antipode(alg, verbose=True))
        with self.subTest(p=p, axiom="evaluation_coevaluation"):
            self.assertTrue(axioms.check_evaluation_coevaluation(alg, verbose=True))

    # p in [2,5] (matches what test_p5 previously validated by hand;
    # RestrictedSl2's dimension is only p^3, so this stays cheap even at
    # the top of the range) -- several independent random draws.
    def test_random_1(self):
        self._check(random_p())

    def test_random_2(self):
        self._check(random_p())

    def test_random_3(self):
        self._check(random_p())

    def test_p2_exhaustive(self):
        # dimension p^3 = 8, cheap enough to check every basis element.
        alg = RestrictedSl2(p=2)
        full_basis = list(alg.basis())
        self.assertTrue(axioms.check_counit(alg, samples=full_basis, verbose=False))
        self.assertTrue(axioms.check_twisted_coassociativity(alg, samples=full_basis, verbose=False))
        self.assertTrue(axioms.check_antipode(alg, samples=full_basis, verbose=False))

    def test_rejects_small_p(self):
        with self.assertRaises(ValueError):
            RestrictedSl2(p=1)

    def test_dimension(self):
        alg = RestrictedSl2(p=3)
        self.assertEqual(len(list(alg.basis())), 3 ** 3)

    def test_trivial_associator(self):
        alg = RestrictedSl2(p=3)
        one = alg.unit()
        self.assertEqual(alg.associator(), alg.associator_inv())
        self.assertEqual(alg.alpha(), one)
        self.assertEqual(alg.beta(), one)

    def test_pretty(self):
        alg = RestrictedSl2(p=3)
        E = Element.basis((1, 0, 0))
        F = Element.basis((0, 1, 0))
        self.assertEqual(alg.pretty(alg.mul(E, F)), "E F")
        self.assertEqual(alg.pretty(alg.unit()), "1")

    def test_generator_accessors(self):
        alg = RestrictedSl2(p=3)
        self.assertEqual(alg.E, alg.elt((1, 0, 0)))
        self.assertEqual(alg.F, alg.elt((0, 1, 0)))
        self.assertEqual(alg.K, alg.elt((0, 0, 1)))
        # tagged, so * works directly
        self.assertEqual(alg.E * alg.F, alg.mul(alg.E, alg.F))

    def test_defining_relations_via_star_and_pow(self):
        # The relations as written in the module docstring, evaluated
        # via E/F/K and */** -- each should come out to exactly 0.
        alg = RestrictedSl2(p=3)
        E, F, K = alg.E, alg.F, alg.K
        q, qinv = alg.q(1), alg.q(-1)
        Kinv = alg.elt((0, 0, alg.p - 1))
        zero = Element.zero()

        self.assertEqual(K * E - q ** 2 * E * K, zero)
        self.assertEqual(K * F - q ** (-2) * F * K, zero)
        self.assertEqual(E * F - F * E - (K - Kinv) * (1 / (q - qinv)), zero)
        self.assertEqual(E ** alg.p, zero)
        self.assertEqual(K ** alg.p - alg.unit(), zero)


def _project_key(key, p):
    a, b, c = key
    return (a, b, c % p)


def _project(elem, p):
    result = Element()
    for key, c in elem.terms.items():
        result.add_term(_project_key(key, p), c)
    return result


class QuotientConsistencyTests(unittest.TestCase):
    """RestrictedSl2 should be exactly QuantumSl2Quasi quotiented by
    K^p = 1 (see restricted_sl2.py's module docstring): the map
    (a, b, c) -> (a, b, c mod p) should be an algebra homomorphism from
    QuantumSl2Quasi's underlying associative algebra onto RestrictedSl2's
    -- checked here exhaustively over every pair of basis elements, for
    every t (multiply_basis doesn't depend on t, so this is checking the
    E/F/K relations themselves, independently of the coalgebra choices
    made in either class)."""

    def _check(self, p, t):
        big = QuantumSl2Quasi(p=p, t=t)
        small = RestrictedSl2(p=p)
        for x in big.basis():
            for y in big.basis():
                lhs = _project(big.mul(Element.basis(x), Element.basis(y)), p)
                rhs = small.mul(Element.basis(_project_key(x, p)), Element.basis(_project_key(y, p)))
                self.assertEqual(lhs, rhs, msg=f"mismatch at p={p}, t={t}, x={x}, y={y}")

    # p in [2,4] (matches what test_p4 previously validated by hand; the
    # exhaustive double loop over the full basis is still fast at that
    # size, ~3s -- see CLAUDE.md's performance notes).
    def test_random_1(self):
        self._check(*random_p_t())

    def test_random_2(self):
        self._check(*random_p_t())


def _project2(elem, p):
    result = Element()
    for key, c in elem.terms.items():
        k1, k2 = key
        result.add_term(TensorKey((_project_key(k1, p), _project_key(k2, p))), c)
    return result


class BraidingTests(unittest.TestCase):
    """R-matrix/monodromy/Drinfeld/ribbon/F-element: cross-checked against
    projecting QuantumSl2Quasi's corresponding elements through the
    K^p = 1 quotient (see restricted_sl2.py's module docstring for the
    derivation of each formula), and checked against the same generic
    axioms as QuantumSl2Quasi's quasi-triangular structure."""

    def _check(self, p, t):
        big = QuantumSl2Quasi(p=p, t=t)
        small = RestrictedSl2(p=p)

        with self.subTest(p=p, t=t, check="r_matrix vs projection"):
            self.assertEqual(small.r_matrix(), _project2(big.r_matrix(), p))
        with self.subTest(p=p, t=t, check="monodromy vs projection"):
            self.assertEqual(small.monodromy(), _project2(big.monodromy(), p))
        with self.subTest(p=p, t=t, check="drinfeld vs projection"):
            self.assertEqual(small.drinfeld(), _project(big.drinfeld(), p))
        with self.subTest(p=p, t=t, check="ribbon vs projection"):
            self.assertEqual(small.ribbon(), _project(big.ribbon(), p))

    # p in [2,3]: p=4 is deliberately excluded here, same reasoning as
    # HexagonTests -- building QuantumSl2Quasi(p=4, ...)'s
    # ribbon/monodromy/drinfeld in Q(zeta_16) just for this cross-check
    # takes ~50s on its own, and p=2,3 already establish the pattern.
    def test_random_1(self):
        self._check(*random_p_t(max_p=3))

    def test_random_2(self):
        self._check(*random_p_t(max_p=3))

    def test_axioms_p2(self):
        alg = RestrictedSl2(p=2)
        self.assertTrue(axioms.check_r_matrix_intertwiner(alg, verbose=True))
        self.assertTrue(axioms.check_hexagon(alg, verbose=True))
        self.assertTrue(axioms.check_ribbon(alg, verbose=True))
        self.assertTrue(axioms.check_s_delta_compatibility(alg, verbose=True))

    def test_axioms_p3(self):
        alg = RestrictedSl2(p=3)
        self.assertTrue(axioms.check_r_matrix_intertwiner(alg, verbose=True))
        self.assertTrue(axioms.check_hexagon(alg, verbose=True))
        self.assertTrue(axioms.check_ribbon(alg, verbose=True))
        self.assertTrue(axioms.check_s_delta_compatibility(alg, verbose=True))

    def test_f_element_trivial(self):
        from hopfsym.element import tensor

        alg = RestrictedSl2(p=3)
        self.assertEqual(alg.f_element(), tensor(alg.unit(), alg.unit()))


if __name__ == "__main__":
    unittest.main()
