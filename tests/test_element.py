"""Tests for Element's optional algebra tag (``.alg``) and the ``*``/``**``
overloads it enables (``E * F`` meaning ``alg.mul(E, F)``, ``E ** n``
meaning repeated ``*``, when the operand(s) are tagged with an algebra)
-- see element.py's Element docstring and algebra.py's
``QuasiHopfAlgebra.elt()``.

Exercised against RestrictedSl2 since it's a small, cheap algebra, but
this is core (element.py/algebra.py) behaviour, not specific to that
example -- every algebra gets ``elt()``/tagged ``mul()`` results for free.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hopfsym.element import Element
from hopfsym.examples import RestrictedSl2


class ElementProductOverloadTests(unittest.TestCase):
    def setUp(self):
        self.alg = RestrictedSl2(p=3)

    def test_tagged_star_matches_mul(self):
        alg = self.alg
        E, F = alg.elt((1, 0, 0)), alg.elt((0, 1, 0))
        self.assertEqual(E * F, alg.mul(E, F))
        self.assertEqual(F * E, alg.mul(F, E))

    def test_chaining_and_arithmetic(self):
        alg = self.alg
        E, F = alg.elt((1, 0, 0)), alg.elt((0, 1, 0))
        # a result of * is itself tagged, so further *, +, -, scalar*
        # keep working without re-tagging.
        self.assertEqual((E * F) * F, alg.mul(alg.mul(E, F), F))
        self.assertEqual(2 * E * F - E, (2 * alg.mul(E, F)) - E)

    def test_untagged_star_still_raises(self):
        a, b = Element.basis((1, 0, 0)), Element.basis((0, 1, 0))
        with self.assertRaises(TypeError):
            a * b

    def test_tagged_times_untagged_uses_the_tag(self):
        # None acts as a wildcard: e.g. alg.unit()/comul()/antipode()
        # results stay untagged, but still combine with a tagged operand.
        alg = self.alg
        E = alg.elt((1, 0, 0))
        self.assertEqual(E * alg.unit(), E)
        self.assertEqual(alg.unit() * E, E)

    def test_mismatched_algebra_instances_raise(self):
        alg2 = RestrictedSl2(p=3)
        E1 = self.alg.elt((1, 0, 0))
        E2 = alg2.elt((1, 0, 0))
        with self.assertRaises(ValueError):
            E1 * E2
        with self.assertRaises(ValueError):
            E1 + E2

    def test_scalar_multiplication_still_works(self):
        alg = self.alg
        E = alg.elt((1, 0, 0))
        self.assertEqual(2 * E, E + E)
        self.assertEqual(E * 2, E + E)


class ElementPowerOverloadTests(unittest.TestCase):
    def setUp(self):
        self.alg = RestrictedSl2(p=3)

    def test_pow_matches_repeated_mul(self):
        E = self.alg.E
        self.assertEqual(E ** 2, E * E)
        self.assertEqual(E ** 3, E * E * E)

    def test_pow_zero_is_unit(self):
        E = self.alg.E
        self.assertEqual(E ** 0, self.alg.unit())

    def test_pow_respects_algebra_relations(self):
        # E^p = 0 is a defining relation of RestrictedSl2.
        E = self.alg.E
        self.assertEqual(E ** self.alg.p, Element.zero())

    def test_pow_untagged_raises(self):
        a = Element.basis((1, 0, 0))
        with self.assertRaises(TypeError):
            a ** 2

    def test_pow_negative_raises(self):
        E = self.alg.E
        with self.assertRaises(ValueError):
            E ** (-1)


if __name__ == "__main__":
    unittest.main()
