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

from hopfsym import axioms
from hopfsym.element import Element, TensorKey
from hopfsym.examples import QuantumSl2Quasi, RestrictedSl2


class RestrictedSl2AxiomTests(unittest.TestCase):
    def _check(self, p):
        alg = RestrictedSl2(p=p)
        with self.subTest(p=p, axiom="associativity"):
            self.assertTrue(axioms.check_associativity(alg, verbose=False))
        with self.subTest(p=p, axiom="bialgebra_homomorphism"):
            self.assertTrue(axioms.check_bialgebra_homomorphism(alg, verbose=False))
        with self.subTest(p=p, axiom="counit"):
            self.assertTrue(axioms.check_counit(alg, verbose=False))
        with self.subTest(p=p, axiom="twisted_coassociativity"):
            # Phi is trivial, so this is ordinary coassociativity.
            self.assertTrue(axioms.check_twisted_coassociativity(alg, verbose=False))
        with self.subTest(p=p, axiom="pentagon"):
            self.assertTrue(axioms.check_pentagon(alg, verbose=False))
        with self.subTest(p=p, axiom="antipode"):
            self.assertTrue(axioms.check_antipode(alg, verbose=False))
        with self.subTest(p=p, axiom="evaluation_coevaluation"):
            self.assertTrue(axioms.check_evaluation_coevaluation(alg, verbose=False))

    def test_p2(self):
        self._check(p=2)

    def test_p3(self):
        self._check(p=3)

    def test_p4(self):
        self._check(p=4)

    def test_p5(self):
        self._check(p=5)

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
                self.assertEqual(lhs, rhs, msg=f"mismatch at x={x}, y={y}")

    def test_p2(self):
        self._check(p=2, t=1)

    def test_p3(self):
        self._check(p=3, t=1)

    def test_p4(self):
        self._check(p=4, t=1)


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

    def test_p2_t1(self):
        self._check(p=2, t=1)

    def test_p2_t3(self):
        self._check(p=2, t=3)

    def test_p3_t1(self):
        self._check(p=3, t=1)

    # p=4 is deliberately not included here: it was checked manually
    # during development (all four projections matched) but building
    # QuantumSl2Quasi(p=4, ...)'s ribbon/monodromy/drinfeld in the
    # Q(zeta_16) field just for this cross-check takes ~50s on its own --
    # not worth paying on every test run given p=2,3 already establish
    # the pattern (same reasoning as HexagonTests dropping p=4).

    def test_axioms_p2(self):
        alg = RestrictedSl2(p=2)
        self.assertTrue(axioms.check_r_matrix_intertwiner(alg, verbose=False))
        self.assertTrue(axioms.check_hexagon(alg, verbose=False))
        self.assertTrue(axioms.check_ribbon(alg, verbose=False))
        self.assertTrue(axioms.check_s_delta_compatibility(alg, verbose=False))

    def test_axioms_p3(self):
        alg = RestrictedSl2(p=3)
        self.assertTrue(axioms.check_r_matrix_intertwiner(alg, verbose=False))
        self.assertTrue(axioms.check_hexagon(alg, verbose=False))
        self.assertTrue(axioms.check_ribbon(alg, verbose=False))
        self.assertTrue(axioms.check_s_delta_compatibility(alg, verbose=False))

    def test_f_element_trivial(self):
        from hopfsym.element import tensor

        alg = RestrictedSl2(p=3)
        self.assertEqual(alg.f_element(), tensor(alg.unit(), alg.unit()))


if __name__ == "__main__":
    unittest.main()
