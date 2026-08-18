"""Standalone sanity checks for the two coefficient rings in
hopfsym.qring: QRational (free q) and CycloNum (q a root of unity).
These don't depend on any particular algebra example.
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hopfsym.qring import CycloNum, QRational, q_power


class QRationalTests(unittest.TestCase):
    def test_basic_arithmetic(self):
        q = q_power(1)
        self.assertEqual(q * q, q_power(2))
        self.assertEqual(q + q, QRational.constant(2) * q)
        self.assertEqual(q - q, QRational.zero())

    def test_inverse(self):
        q = q_power(1)
        inv = QRational.one() / q
        self.assertEqual(inv, q_power(-1))
        self.assertEqual(q * inv, QRational.one())

    def test_division_with_non_monomial_denominator(self):
        q = q_power(1)
        denom = q - q_power(-1)
        x = QRational.one() / denom
        self.assertEqual(x * denom, QRational.one())

    def test_q_is_genuinely_free(self):
        # Nothing should force any relation among powers of q: q^5 is
        # not equal to q^0, q^1, ... for a free variable.
        for k in range(1, 6):
            self.assertNotEqual(q_power(5), q_power(k % 5))
        self.assertNotEqual(q_power(3), QRational.one())

    def test_mixing_with_plain_numbers(self):
        q = q_power(1)
        self.assertEqual(1 + q - q, QRational.one())
        self.assertEqual(Fraction(1, 2) * (q + q), q)


class CycloNumTests(unittest.TestCase):
    def test_primitive_root_of_unity(self):
        n = 6
        zeta = CycloNum.power(n, 1)
        self.assertEqual(zeta ** n, CycloNum.one(n))
        for k in range(1, n):
            self.assertNotEqual(zeta ** k, CycloNum.one(n), msg=f"zeta^{k} should not be 1")

    def test_minimal_polynomial_reduction(self):
        # Phi_6(x) = x^2 - x + 1, so zeta^2 = zeta - 1 in Q(zeta_6).
        n = 6
        zeta = CycloNum.power(n, 1)
        self.assertEqual(zeta ** 2, zeta - CycloNum.one(n))

    def test_inverse_of_q_minus_qinv(self):
        # This is exactly the quantity the quantum-group commutator
        # [E, F] divides by; it must be nonzero and invertible whenever
        # n > 2 (so that zeta != +-1).
        for n in (4, 6, 8, 10):
            zeta = CycloNum.power(n, 1)
            denom = zeta - zeta ** (-1)
            self.assertFalse(denom.is_zero())
            inv = denom.inverse()
            self.assertEqual(denom * inv, CycloNum.one(n))

    def test_mixing_with_plain_numbers(self):
        n = 6
        zeta = CycloNum.power(n, 1)
        self.assertEqual(1 + zeta - zeta, CycloNum.one(n))
        self.assertEqual(Fraction(1, 2) * (zeta + zeta), zeta)

    def test_mismatched_field_degree_rejected(self):
        a = CycloNum.power(6, 1)
        b = CycloNum.power(4, 1)
        with self.assertRaises(ValueError):
            _ = a + b


if __name__ == "__main__":
    unittest.main()
