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

from hopfsym.element import Element, tensor, tensor_mul, Δ, ε
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


class TensorElementProductTests(unittest.TestCase):
    """`*` on two n-fold tensor Elements means the componentwise
    tensor-algebra product, (a(x)b)*(c(x)d) = (a*c)(x)(b*d) -- i.e.
    tensor_mul, not alg.mul/multiply_basis (which only knows plain,
    single-factor basis keys)."""

    def setUp(self):
        self.alg = RestrictedSl2(p=3)

    def test_tensor_propagates_tag(self):
        alg = self.alg
        E, F = alg.E, alg.F
        self.assertIs(tensor(E, F).alg, alg)
        # None is a wildcard (same convention as +/-): one tagged operand
        # is enough for the tensor result to come out tagged too.
        self.assertIs(tensor(Element.basis((1, 0, 0)), F).alg, alg)
        self.assertIs(tensor(Element.basis((1, 0, 0)), Element.basis((0, 1, 0))).alg, None)

    def test_star_matches_tensor_mul(self):
        alg = self.alg
        E, F, K = alg.E, alg.F, alg.K
        left = tensor(E, F)
        right = tensor(K, E)
        self.assertEqual(left * right, tensor_mul(alg, left, right))
        self.assertEqual(left * right, tensor(E * K, F * E))

    def test_result_is_tagged_and_chains(self):
        alg = self.alg
        E, F, K = alg.E, alg.F, alg.K
        prod = tensor(E, F) * tensor(K, E)
        self.assertIs(prod.alg, alg)
        # chains with a further tensor product without re-tagging
        self.assertEqual(prod * tensor(E, F), tensor_mul(alg, prod, tensor(E, F)))

    def test_mismatched_arity_raises(self):
        alg = self.alg
        E, F = alg.E, alg.F
        with self.assertRaises(ValueError):
            E * tensor(E, F)

    def test_untagged_tensor_star_still_raises(self):
        a = tensor(Element.basis((1, 0, 0)), Element.basis((0, 1, 0)))
        with self.assertRaises(TypeError):
            a * a


class DeltaDispatchTests(unittest.TestCase):
    """Δ(x) dispatches to x.alg.comul(x) -- which algebra's coproduct
    rule applies is read off the tagged argument itself."""

    def setUp(self):
        self.alg = RestrictedSl2(p=3)

    def test_delta_matches_comul(self):
        alg = self.alg
        E = alg.E
        self.assertEqual(Δ(E), alg.comul(E))
        self.assertEqual(Δ(alg.K), alg.comul(alg.K))

    def test_delta_untagged_raises(self):
        a = Element.basis((1, 0, 0))
        with self.assertRaises(TypeError):
            Δ(a)

    def test_delta_picks_up_the_right_algebra(self):
        # Two different algebra instances -> two different comul rules;
        # Δ must use whichever .alg the argument is actually tagged with,
        # not e.g. a module-global default.
        alg2 = RestrictedSl2(p=5)
        self.assertEqual(Δ(self.alg.E), self.alg.comul(self.alg.E))
        self.assertEqual(Δ(alg2.E), alg2.comul(alg2.E))


class EpsilonDispatchTests(unittest.TestCase):
    """ε(x) dispatches to x.alg.counit(x) -- same idea as Δ, for the
    counit (a scalar, not an Element) instead of the coproduct."""

    def setUp(self):
        self.alg = RestrictedSl2(p=3)

    def test_epsilon_matches_counit(self):
        alg = self.alg
        self.assertEqual(ε(alg.E), alg.counit(alg.E))
        self.assertEqual(ε(alg.K), alg.counit(alg.K))

    def test_epsilon_is_an_algebra_homomorphism_on_generators(self):
        # epsilon(1) = 1, epsilon(E) = epsilon(F) = 0 (E, F augmentation
        # ideal), epsilon(K) = 1 -- direct sanity check independent of
        # axioms.check_counit's own (randomised) sampling.
        alg = self.alg
        self.assertEqual(ε(alg.E), 0)
        self.assertEqual(ε(alg.F), 0)
        self.assertEqual(ε(alg.K), 1)

    def test_epsilon_untagged_raises(self):
        a = Element.basis((1, 0, 0))
        with self.assertRaises(TypeError):
            ε(a)

    def test_epsilon_picks_up_the_right_algebra(self):
        alg2 = RestrictedSl2(p=5)
        self.assertEqual(ε(self.alg.E), self.alg.counit(self.alg.E))
        self.assertEqual(ε(alg2.E), alg2.counit(alg2.E))


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

    def test_pow_on_tensor_element(self):
        alg = self.alg
        K = alg.K
        KK = tensor(K, K)  # tagged, arity 2
        self.assertEqual(KK ** 0, tensor(alg.unit(), alg.unit()))
        self.assertEqual(KK ** 2, KK * KK)
        self.assertEqual(KK ** alg.p, tensor(alg.unit(), alg.unit()))  # K^p = 1


if __name__ == "__main__":
    unittest.main()
