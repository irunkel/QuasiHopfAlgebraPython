"""Shared helpers for randomising *which algebra instance* (p, t, N,
beta_power) a test runs against -- the model-choice analogue of
hopfsym.axioms's random sample-seed handling, which randomises *which
basis elements* get sampled within a fixed instance.

Not a test module itself (no test_ prefix, so unittest discovery skips
it); imported by the test_*.py files that need it.

Bounds are picked empirically for reasonable test-suite runtime, not the
largest values these algebras actually support -- see CLAUDE.md's
performance notes for where each bound came from (e.g. hexagon/R-matrix
checks at p=4 have been individually timed at tens of seconds, which is
why they stay excluded here even though check_all alone is fast at that
size). Uses the plain top-level `random` module (auto-seeded from OS
entropy at process start), not a tracked seed: unlike axioms.py's sample
draws, there's nothing to "reproduce" here beyond the actual p/t/N
values themselves, which every caller reports via unittest's `subTest`
(shown automatically on failure) -- see CLAUDE.md.
"""

from __future__ import annotations

import random


def random_p(min_p: int = 2, max_p: int = 5) -> int:
    """A random p in [min_p, max_p] (for RestrictedSl2, which has no t)."""
    return random.randint(min_p, max_p)


def random_p_t(max_p: int = 4) -> tuple:
    """A random (p, t) for QuantumSl2Quasi (or a QuantumSl2Quasi used
    alongside a RestrictedSl2 for cross-checking): p in [2, max_p], t odd
    in [1, 2p-1]."""
    p = random.randint(2, max_p)
    t = random.choice(range(1, 2 * p, 2))
    return p, t


def valid_beta_powers(N: int) -> list:
    """The four beta_power residues mod 8 valid for this N (beta^4 =
    (-1)^N, i.e. beta_power % 2 == N % 2) -- SymplecticFermionQ's "four
    possible choices of beta" for given N."""
    return [b for b in range(8) if b % 2 == N % 2]


def random_N(max_N: int = 3) -> int:
    """A random N in [1, max_N] for SymplecticFermionQ."""
    return random.randint(1, max_N)


def random_N_beta_power(max_N: int = 3) -> tuple:
    """A random (N, beta_power) for SymplecticFermionQ: N in [1, max_N],
    beta_power one of the four residues mod 8 of the right parity for
    that N (beta^4 = (-1)^N)."""
    N = random_N(max_N)
    return N, random.choice(valid_beta_powers(N))
