"""Regression tests for the U_q^{(Phi)}sl(2) example, checking it
against every required (non-optional) quasi-Hopf axiom in
hopfsym.axioms, for randomly chosen (p, t) (see tests/_random_model.py --
a fresh draw every run, not the same fixed case every time; on failure,
unittest's subTest reports which (p, t) was used, and axioms.py's checks
themselves report the random sample seed). These mirror what ``testall``
does in the original Mathematica implementation this was ported from,
for the core algebra; the R-matrix/ribbon/hexagon/Drinfeld/F-element
checks (optional quasi-triangular structure) are in
test_quantum_sl2_quasi_braiding.py instead.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hopfsym import axioms
from hopfsym.examples import QuantumSl2Quasi

from _random_model import random_p_t


class QuantumSl2QuasiAxiomTests(unittest.TestCase):
    def _check(self, p, t):
        alg = QuantumSl2Quasi(p=p, t=t)
        with self.subTest(p=p, t=t, axiom="associativity"):
            self.assertTrue(axioms.check_associativity(alg, verbose=True))
        with self.subTest(p=p, t=t, axiom="bialgebra_homomorphism"):
            self.assertTrue(axioms.check_bialgebra_homomorphism(alg, verbose=True))
        with self.subTest(p=p, t=t, axiom="counit"):
            self.assertTrue(axioms.check_counit(alg, verbose=True))
        with self.subTest(p=p, t=t, axiom="twisted_coassociativity"):
            self.assertTrue(axioms.check_twisted_coassociativity(alg, verbose=True))
        with self.subTest(p=p, t=t, axiom="pentagon"):
            self.assertTrue(axioms.check_pentagon(alg, verbose=True))
        with self.subTest(p=p, t=t, axiom="antipode"):
            self.assertTrue(axioms.check_antipode(alg, verbose=True))

    # Each of these draws its own random (p, t) (p in [2, 4], matching
    # what test_p4_t1 previously validated by hand) -- several independent
    # methods rather than one loop, so a failure in one doesn't stop the
    # others from running and reporting their own (p, t).
    def test_random_1(self):
        self._check(*random_p_t())

    def test_random_2(self):
        self._check(*random_p_t())

    def test_random_3(self):
        self._check(*random_p_t())

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
