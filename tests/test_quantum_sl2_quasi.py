"""Regression tests for the U_q^{(Phi)}sl(2) example, checking it
against every generic quasi-Hopf axiom in hopfsym.axioms, for a few
small (p, t). These mirror what ``testall`` does in the original
Mathematica code (reference/hopf-Uqsl2-quasi.txt), minus the R-matrix /
ribbon / hexagon checks, which are not yet ported (see CLAUDE.md).
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hopfsym import axioms
from hopfsym.examples import QuantumSl2Quasi


class QuantumSl2QuasiAxiomTests(unittest.TestCase):
    def _check(self, p, t):
        alg = QuantumSl2Quasi(p=p, t=t)
        with self.subTest(p=p, t=t, axiom="associativity"):
            self.assertTrue(axioms.check_associativity(alg, verbose=False))
        with self.subTest(p=p, t=t, axiom="bialgebra_homomorphism"):
            self.assertTrue(axioms.check_bialgebra_homomorphism(alg, verbose=False))
        with self.subTest(p=p, t=t, axiom="counit"):
            self.assertTrue(axioms.check_counit(alg, verbose=False))
        with self.subTest(p=p, t=t, axiom="twisted_coassociativity"):
            self.assertTrue(axioms.check_twisted_coassociativity(alg, verbose=False))
        with self.subTest(p=p, t=t, axiom="pentagon"):
            self.assertTrue(axioms.check_pentagon(alg, verbose=False))
        with self.subTest(p=p, t=t, axiom="antipode"):
            self.assertTrue(axioms.check_antipode(alg, verbose=False))

    def test_p2_t1(self):
        self._check(p=2, t=1)

    def test_p3_t1(self):
        self._check(p=3, t=1)

    def test_p3_t3(self):
        self._check(p=3, t=3)

    def test_p2_t3(self):
        self._check(p=2, t=3)

    def test_p4_t1(self):
        # p=4 has dimension 2*4^3=128; cheap enough to include here now
        # that axioms.py samples randomly by default instead of
        # exhaustively (see CLAUDE.md's performance notes).
        self._check(p=4, t=1)

    def test_p2_exhaustive(self):
        # A small enough algebra (dimension 16) that checking every
        # single basis element, not just a random sample, is still fast
        # -- a stronger regression check than the sampled default.
        alg = QuantumSl2Quasi(p=2, t=1)
        full_basis = list(alg.basis())
        self.assertTrue(axioms.check_counit(alg, samples=full_basis, verbose=False))
        self.assertTrue(
            axioms.check_twisted_coassociativity(alg, samples=full_basis, verbose=False)
        )
        self.assertTrue(axioms.check_antipode(alg, samples=full_basis, verbose=False))

    def test_rejects_even_t(self):
        with self.assertRaises(ValueError):
            QuantumSl2Quasi(p=2, t=2)

    def test_rejects_small_p(self):
        with self.assertRaises(ValueError):
            QuantumSl2Quasi(p=1, t=1)

    def test_dimension(self):
        alg = QuantumSl2Quasi(p=3, t=1)
        self.assertEqual(len(list(alg.basis())), 2 * 3 ** 3)


if __name__ == "__main__":
    unittest.main()
