"""Regression tests for the symplectic fermion quasi-Hopf algebra
Q(N, beta) -- see symplectic_fermion.py's module docstring for the
definition and for why verification here leans on two facts given
explicitly in the paper (arXiv:1706.08164, Remark right after
eq:Q-antipode-def), rather than a Mathematica reference (there isn't
one for this algebra).

Only the quasi-Hopf data is tested here (no R-matrix/ribbon yet).
"""

import sys
import unittest
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hopfsym import axioms
from hopfsym.element import Element, tensor, tensor_mul, apply_to_factor
from hopfsym.examples import SymplecticFermionQ

from _random_model import random_N, valid_beta_powers


class BasicRelationsTests(unittest.TestCase):
    def test_dimension(self):
        alg = SymplecticFermionQ(N=2, beta_power=0)
        self.assertEqual(len(list(alg.basis())), 2 ** (2 * 2 + 2))

    def test_rejects_small_N(self):
        with self.assertRaises(ValueError):
            SymplecticFermionQ(N=0, beta_power=0)

    def test_rejects_wrong_parity_beta(self):
        with self.assertRaises(ValueError):
            SymplecticFermionQ(N=1, beta_power=0)  # N odd needs odd beta_power
        with self.assertRaises(ValueError):
            SymplecticFermionQ(N=2, beta_power=1)  # N even needs even beta_power

    def test_anticommutation_relations(self):
        alg = SymplecticFermionQ(N=2, beta_power=0)
        K = alg._K(1)
        f1p, f1m, f2p = alg._f(0), alg._f(1), alg._f(2)
        e1 = alg.e1

        with self.subTest("{f,K}=0"):
            self.assertEqual(alg.mul(f1p, K) + alg.mul(K, f1p), Element.zero())
        with self.subTest("{f1+,f1-}=e1"):
            self.assertEqual(alg.mul(f1p, f1m) + alg.mul(f1m, f1p), e1)
        with self.subTest("{f1+,f2+}=0"):
            self.assertEqual(alg.mul(f1p, f2p) + alg.mul(f2p, f1p), Element.zero())
        with self.subTest("(f1+)^2=0"):
            self.assertEqual(alg.mul(f1p, f1p), Element.zero())
        with self.subTest("K^4=1"):
            K4 = alg.mul(alg.mul(alg.mul(K, K), K), K)
            self.assertEqual(K4, alg.unit())


class AxiomTests(unittest.TestCase):
    def _check(self, N, beta_power):
        alg = SymplecticFermionQ(N=N, beta_power=beta_power)
        with self.subTest(N=N, beta_power=beta_power):
            # verbose=False: check_all's own PASS/FAIL-per-axiom summary
            # is tied to this flag (unlike the individual check_* below,
            # which only ever print on failure); the random sample seed
            # still gets reported unconditionally on failure regardless
            # (see axioms._report_seed).
            self.assertTrue(axioms.check_all(alg, verbose=False))

    def _check_beta_sweep(self):
        # A random N (in [1,3], matching what test_N3 previously
        # validated by hand -- dimension 2^(2N+2) grows fast), with
        # *every* valid beta_power for that N exhaustively (only 4
        # values, cheap, and worth keeping exhaustive rather than
        # sampled -- unlike which N gets picked).
        N = random_N()
        for bp in valid_beta_powers(N):
            self._check(N=N, beta_power=bp)

    def test_random_beta_sweep_1(self):
        self._check_beta_sweep()

    def test_random_beta_sweep_2(self):
        self._check_beta_sweep()

    def test_phi_inverse(self):
        # mu(Phi, Phi^-1) == 1(x)1(x)1 -- not part of the generic checks,
        # but a direct sanity check worth pinning down as a regression.
        N = random_N()
        for bp in valid_beta_powers(N):
            alg = SymplecticFermionQ(N=N, beta_power=bp)
            one = alg.unit()
            identity3 = tensor(tensor(one, one), one)
            prod = tensor_mul(alg, alg.associator(), alg.associator_inv())
            with self.subTest(N=N, beta_power=bp):
                self.assertEqual(prod, identity3)


class PaperRemarkTests(unittest.TestCase):
    """The remark right after eq:Q-antipode-def in the paper gives two
    facts we can check directly against the paper's own formulas,
    independent of the generic axiom machinery."""

    def test_even_N_beta_one_gives_honest_hopf_algebra(self):
        # "For even N and beta^2=1 we have Salpha=Sbeta=1 and
        # Phi=1(x)1(x)1."
        for N in (2, 4):
            alg = SymplecticFermionQ(N=N, beta_power=0)  # beta=1
            one = alg.unit()
            with self.subTest(N=N):
                self.assertEqual(alg.alpha(), one)
                self.assertEqual(alg.beta(), one)
                self.assertEqual(alg.associator(), tensor(tensor(one, one), one))

    def test_explicit_twisted_coassociativity_formula_for_fminus(self):
        # The paper spells out (Delta x id)(Delta(f_i^-)) and
        # (id x Delta)(Delta(f_i^-)) explicitly for general N; comparing
        # against these checks comul() in isolation (independent of
        # whether associator()/antipode() also happen to be right).
        for N, bp in [(1, 1), (2, 0), (2, 2), (3, 1), (4, 0)]:
            alg = SymplecticFermionQ(N=N, beta_power=bp)
            one = alg.unit()
            e0, e1 = alg.e0, alg.e1
            K = alg._K(1)
            i_ = alg.i()
            fm = alg._f(1)  # f_1^-
            omega_minus = alg._omega(-1)
            KK = tensor(K, K)

            bracket_left = (
                tensor(e0, e0)
                + (-i_) * tensor(e0, e1)
                + (-i_) * tensor(e1, e0)
                + (-Fraction((-1) ** N)) * tensor(e1, e1)
            )
            bracket_right = (
                tensor(e0, e0) + (-i_) * tensor(e0, e1) + (-i_) * tensor(e1, e0) + Fraction(-1) * tensor(e1, e1)
            )

            expected_delta_id = (
                tensor(fm, tensor(one, one))
                + tensor(omega_minus, tensor(fm, one))
                + tensor(tensor_mul(alg, KK, bracket_left), fm)
            )
            expected_id_delta = (
                tensor(fm, tensor(one, one))
                + tensor(omega_minus, tensor(fm, one))
                + tensor(tensor_mul(alg, KK, bracket_right), fm)
            )

            actual_delta_id = apply_to_factor(alg.comul(fm), 0, alg.comul)
            actual_id_delta = apply_to_factor(alg.comul(fm), 1, alg.comul)

            with self.subTest(N=N, beta_power=bp, which="(Delta x id)Delta(f-)"):
                self.assertEqual(actual_delta_id, expected_delta_id)
            with self.subTest(N=N, beta_power=bp, which="(id x Delta)Delta(f-)"):
                self.assertEqual(actual_id_delta, expected_id_delta)


class GeneratorAccessorTests(unittest.TestCase):
    def test_K_matches_internal_helper_and_is_tagged(self):
        alg = SymplecticFermionQ(N=2, beta_power=0)
        self.assertEqual(alg.K, alg._K(1))
        self.assertEqual(alg.K * alg.K, alg.mul(alg.K, alg.K))

    def test_f_matches_internal_helper_and_is_tagged(self):
        alg = SymplecticFermionQ(N=2, beta_power=0)
        # f(1,'+') -> pos 0, f(1,'-') -> pos 1, f(2,'+') -> pos 2, ...
        self.assertEqual(alg.f(1, "+"), alg._f(0))
        self.assertEqual(alg.f(1, "-"), alg._f(1))
        self.assertEqual(alg.f(2, "+"), alg._f(2))
        self.assertEqual(alg.f(2, "-"), alg._f(3))
        self.assertEqual(alg.f(1, "+") * alg.f(1, "-"), alg.mul(alg._f(0), alg._f(1)))

    def test_f_validates_arguments(self):
        alg = SymplecticFermionQ(N=2, beta_power=0)
        with self.assertRaises(ValueError):
            alg.f(1, "x")
        with self.assertRaises(ValueError):
            alg.f(0, "+")
        with self.assertRaises(ValueError):
            alg.f(3, "+")  # N=2, so i must be 1 or 2

    def test_e0_e1_match_old_formula_and_are_tagged(self):
        alg = SymplecticFermionQ(N=2, beta_power=0)
        one, K2 = alg.unit(), alg._K(2)
        self.assertEqual(alg.e0, Fraction(1, 2) * (one + K2))
        self.assertEqual(alg.e1, Fraction(1, 2) * (one - K2))
        self.assertEqual(alg.e0 * alg.e1, alg.mul(alg.e0, alg.e1))

    def test_defining_relations_via_star_and_pow(self):
        # The relations as written in the module docstring, evaluated
        # via K/f/e0/e1 and */** -- each should come out to exactly 0.
        alg = SymplecticFermionQ(N=2, beta_power=0)
        K = alg.K
        f1p, f1m, f2p = alg.f(1, "+"), alg.f(1, "-"), alg.f(2, "+")
        zero = Element.zero()

        self.assertEqual(f1p * K + K * f1p, zero)  # {f_i^+-, K} = 0
        self.assertEqual(f1p * f1m + f1m * f1p - alg.e1, zero)  # {f_i^+,f_j^-}=delta_ij*e1, i=j
        self.assertEqual(f1p * f2p + f2p * f1p, zero)  # {f_i^eps,f_j^eps}=0, i!=j
        self.assertEqual(f1p * f1p, zero)  # (f_i^eps)^2 = 0
        self.assertEqual(K ** 4 - alg.unit(), zero)  # K^4 = 1


class PrettyTests(unittest.TestCase):
    def test_pretty(self):
        alg = SymplecticFermionQ(N=2, beta_power=0)
        f1p, f1m = alg._f(0), alg._f(1)
        self.assertEqual(alg.pretty(alg.mul(f1p, f1m)), "f₁⁺ f₁⁻")
        self.assertEqual(alg.pretty(alg.unit()), "1")


if __name__ == "__main__":
    unittest.main()
