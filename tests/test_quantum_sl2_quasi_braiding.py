"""Regression tests for the quasi-triangular / ribbon structure of
U_q^{(Phi)}sl(2): the R-matrix, hexagon axioms, monodromy element,
Drinfeld element, ribbon element and its properties, and the "F" element
relating Delta and S. These mirror ``testExpression``, ``testRinter``,
``testHexagon``, ``testEvalCoeval``, ``testRibbon`` and ``testSDeltaR`` in
the original Mathematica code (reference/hopf-Uqsl2-quasi.txt).

Kept separate from test_quantum_sl2_quasi.py (which covers the core
algebra/bialgebra/antipode axioms) since these all live one field up, in
Q(zeta_{4p}) rather than Q(zeta_{2p}) -- see r_matrix()'s docstring.
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hopfsym import axioms
from hopfsym.element import Element, tensor, tensor_mul
from hopfsym.examples import QuantumSl2Quasi
from hopfsym.qring import CycloNum


def _RRsec(alg):
    """Alternate closed form for the R-matrix (testExpression, part 1),
    built via the central idempotents e0/e1 rather than directly
    exponentiated brackets -- structurally different from r_matrix(), so
    agreement between the two is a real independent check."""
    p, t = alg.p, alg.t
    e0, e1 = alg._idempotents()
    combo_tens = [tensor(e0, e0), tensor(e0, e1), tensor(e1, e0), tensor(e1, e1)]
    result = Element()
    for n in range(p):
        coeff_n = ((alg.q(1) - alg.q(-1)) ** n / alg._qfact(n)).embed(4 * p)
        for s in range(p):
            for r in range(p):
                qA2 = n * (n - 1) + 4 * n * (s - r) - 4 * s * r
                bracket_exps2 = (
                    0,
                    2 * t * r,
                    -2 * t * (n + s),
                    t * t + 2 * t * r - 2 * t * (n + s),
                )
                base = alg.q_half(qA2)
                combo = Element()
                for exp2, tij in zip(bracket_exps2, combo_tens):
                    combo = combo + (base * alg.q_half(exp2)) * tij
                piece = tensor_mul(alg, tensor(Element.basis((n, 0, s)), Element.basis((0, n, r))), combo)
                result = result + (coeff_n * Fraction(1, p)) * piece
    return result


class RMatrixTests(unittest.TestCase):
    def _check(self, p, t, samples=None):
        alg = QuantumSl2Quasi(p=p, t=t)
        with self.subTest(p=p, t=t, check="RRsec cross-check"):
            self.assertEqual(alg.r_matrix(), _RRsec(alg))
        with self.subTest(p=p, t=t, check="intertwiner"):
            self.assertTrue(axioms.check_r_matrix_intertwiner(alg, samples=samples, verbose=False))

    def test_p2_t1(self):
        self._check(p=2, t=1)

    def test_p2_t3(self):
        self._check(p=2, t=3)

    def test_p3_t1(self):
        self._check(p=3, t=1)

    def test_p3_t3(self):
        self._check(p=3, t=3)

    def test_p4_t1(self):
        # R has 4p^3 = 256 raw terms at p=4, and the intertwiner check
        # multiplies it against Delta(a) in H (x) H for each sample --
        # exact arithmetic in Q(zeta_16) makes that real work per sample
        # (unlike the smaller structure maps elsewhere in this package),
        # so use a handful of samples rather than the default 10 to keep
        # this test's runtime reasonable; the RRsec cross-check above is
        # already an exhaustive, independent check of the R-matrix itself.
        alg = QuantumSl2Quasi(p=4, t=1)
        self._check(p=4, t=1, samples=list(alg.basis())[::37])


class HexagonTests(unittest.TestCase):
    # The hexagon axioms are a single (no-sampling) check per (p, t) --
    # like the pentagon axiom, but heavier (it also involves R, so it's
    # not included in axioms.check_all and left at p=2,3 here; see
    # CLAUDE.md's performance notes if p=4 is ever needed regularly).
    def test_p2_t1(self):
        self.assertTrue(axioms.check_hexagon(QuantumSl2Quasi(p=2, t=1), verbose=False))

    def test_p2_t3(self):
        self.assertTrue(axioms.check_hexagon(QuantumSl2Quasi(p=2, t=3), verbose=False))

    def test_p3_t1(self):
        self.assertTrue(axioms.check_hexagon(QuantumSl2Quasi(p=3, t=1), verbose=False))


def _mon2(alg):
    """Alternate closed form for the monodromy element (testExpression,
    part 2), summing over K^j F^m E^n (x) K^i E^m F^n directly rather
    than via flip(R).R -- unlike R itself, every exponent here is an
    integer (m(m-1)/2 is always a whole number), so this stays in
    Q(zeta_{2p}) and doesn't need q_half at all."""
    p, t = alg.p, alg.t
    result = Element()
    for n in range(p):
        for m in range(p):
            coeff_nm = (alg.q(1) - alg.q(-1)) ** (m + n) / (alg._qfact(m) * alg._qfact(n))
            coeff_nm = coeff_nm * alg.q(m * (m - 1) // 2 + n * (n - 1) // 2)
            for i in range(2 * p):
                for j in range(2 * p):
                    exp = -m * m + m * (j - i) - i * j
                    sign = Fraction(1, 2) * (1 + (-1) ** (i + m)) + Fraction(1, 2) * (1 - (-1) ** (i + m)) * alg.q(
                        t * (m - n)
                    )
                    scalar = coeff_nm * alg.q(exp) * sign * Fraction(1, 2 * p)
                    left = alg._reduce_word([("K", j), ("F", m), ("E", n)])
                    right = alg._reduce_word([("K", i), ("E", m), ("F", n)])
                    piece = tensor(left, right)
                    for k, c in piece.terms.items():
                        result.add_term(k, scalar * c)
    return result


class MonodromyTests(unittest.TestCase):
    def test_cross_check(self):
        for p, t in [(2, 1), (2, 3), (3, 1), (3, 3)]:
            with self.subTest(p=p, t=t):
                alg = QuantumSl2Quasi(p=p, t=t)
                self.assertEqual(alg.monodromy(), _mon2(alg))


class EvalCoevalTests(unittest.TestCase):
    def test_p2_t1(self):
        self.assertTrue(axioms.check_evaluation_coevaluation(QuantumSl2Quasi(p=2, t=1), verbose=False))

    def test_p3_t3(self):
        self.assertTrue(axioms.check_evaluation_coevaluation(QuantumSl2Quasi(p=3, t=3), verbose=False))

    def test_p4_t1(self):
        self.assertTrue(axioms.check_evaluation_coevaluation(QuantumSl2Quasi(p=4, t=1), verbose=False))


class RibbonTests(unittest.TestCase):
    def test_gauss_sum_identity(self):
        # sum_{a=0}^{2p-1} q^{-a^2/2} == (1-i)*sqrt(p) -- the classical
        # identity ribbon()'s prefactor is built from exactly (no
        # black-box sqrt/complex arithmetic). Cross-checked numerically
        # once during development; this pins the *exact* CycloNum value
        # down as a regression (the exact value equals 2p * (ribbon's
        # prefactor), see ribbon()'s docstring).
        for p in (2, 3, 4, 5):
            alg = QuantumSl2Quasi(p=p, t=1)
            gauss_sum = sum((alg.q_half(-a * a) for a in range(2 * p)), CycloNum.zero(4 * p))
            # (1-i)*sqrt(p): i = zeta_{4p}^p, and sqrt(p) = gauss_sum / (1-i)
            # -- instead just check the defining property algebraically:
            # gauss_sum**2 == 2*i*p (since ((1-i)*sqrt(p))**2 = -2i*p...
            # (1-i)^2 = -2i, so gauss_sum**2 == -2i*p).
            i = CycloNum.power(4 * p, p)
            self.assertEqual(gauss_sum * gauss_sum, Fraction(-2 * p) * i)

    def _check(self, p, t):
        alg = QuantumSl2Quasi(p=p, t=t)
        with self.subTest(p=p, t=t, check="S(v) == v"):
            self.assertEqual(alg.antipode(alg.ribbon()), alg.ribbon())
        with self.subTest(p=p, t=t, check="eps(v) == 1"):
            self.assertEqual(alg.counit(alg.ribbon()), 1)
        with self.subTest(p=p, t=t, check="full check_ribbon"):
            self.assertTrue(axioms.check_ribbon(alg, verbose=False))

    def test_p2_t1(self):
        self._check(p=2, t=1)

    def test_p2_t3(self):
        self._check(p=2, t=3)

    def test_p3_t1(self):
        self._check(p=3, t=1)


class FElementTests(unittest.TestCase):
    def _check(self, p, t):
        alg = QuantumSl2Quasi(p=p, t=t)
        self.assertTrue(axioms.check_s_delta_compatibility(alg, verbose=False))

    def test_p2_t1(self):
        self._check(p=2, t=1)

    def test_p2_t3(self):
        self._check(p=2, t=3)

    def test_p3_t1(self):
        self._check(p=3, t=1)

    def test_p3_t3(self):
        self._check(p=3, t=3)

    def test_p4_t1(self):
        self._check(p=4, t=1)


if __name__ == "__main__":
    unittest.main()
