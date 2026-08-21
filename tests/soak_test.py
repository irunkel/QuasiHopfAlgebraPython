"""Long-running stress test: repeatedly builds fresh random instances of
every example algebra (the same randomisation as tests/_random_model.py
and the test_*.py files -- same bounds, same helpers) and runs every
axiom check against each, plus (for RestrictedSl2/SymplecticFermionQ)
the independent closed-form checks on drinfeld()/f_element() from
tests/_special_elements.py -- the same comparisons test_restricted_sl2.py/
test_symplectic_fermion.py make, reused here rather than duplicated.
Cycles through QuantumSl2Quasi -> RestrictedSl2 -> SymplecticFermionQ
round-robin, for a configurable time budget (default one hour). The
point is to exercise far more of the (p, t, N, beta_power) x
random-basis-sample space than the regular, fast test suite does in one
run -- useful for shaking out rare failures before a release, or after a
change to shared machinery (element.py, algebra.py, axioms.py) that
every example depends on.

Not a unittest module (no test_ prefix, so `python3 -m unittest discover
-s tests` skips it) -- meant to be started directly from the shell::

    python3 tests/soak_test.py                    # run for 1 hour
    python3 tests/soak_test.py --minutes 10        # shorter run
    python3 tests/soak_test.py --sample-size 6     # axioms.py's own default instead

Every check is run with verbose=True, so each one's own PASS/FAIL (for
check_all) or failure diagnostics + the sample seed (for the individual
checks, which -- unlike check_all -- only ever print on failure, see
axioms.py's docstring) show up inline as they happen; nothing is
buffered or re-run after the fact. Stops at the first failing check
(exit code 1) -- the model's exact constructor arguments are printed
right above every round, so a failure is directly reproducible by
rebuilding that one instance. If the whole time budget elapses with no
failures, prints a summary and exits 0. Ctrl-C stops early with the same
summary.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hopfsym import axioms
from hopfsym.examples import QuantumSl2Quasi, RestrictedSl2, SymplecticFermionQ

from _random_model import random_p, random_p_t, random_N_beta_power
from _special_elements import (
    restricted_sl2_expected_drinfeld,
    restricted_sl2_expected_f_element,
    symplectic_fermion_expected_drinfeld,
    symplectic_fermion_expected_f_element,
)


def _make_explicit_formula_check(name, expected_fn):
    """Wrap an expected-value function from _special_elements.py (an
    independent closed form for drinfeld()/f_element(), available for
    some algebras -- see that module's docstring) into a check matching
    axioms.py's check_*(alg, verbose=True) -> bool signature, so it can
    sit in a round's checks list alongside the generic axiom checks."""

    def check(alg, verbose=True) -> bool:
        actual = getattr(alg, name)()
        expected = expected_fn(alg)
        ok = actual == expected
        if not ok and verbose:
            print(f"  {name}() does not match its independent closed form")
            print(f"    {name}() = {actual}")
            print(f"    closed form = {expected}")
        return ok

    check.__name__ = f"check_{name}_matches_explicit_formula"
    return check


_check_restricted_sl2_drinfeld = _make_explicit_formula_check("drinfeld", restricted_sl2_expected_drinfeld)
_check_restricted_sl2_f_element = _make_explicit_formula_check("f_element", restricted_sl2_expected_f_element)
_check_symplectic_fermion_drinfeld = _make_explicit_formula_check("drinfeld", symplectic_fermion_expected_drinfeld)
_check_symplectic_fermion_f_element = _make_explicit_formula_check("f_element", symplectic_fermion_expected_f_element)


def _round_quantum_sl2_quasi():
    p, t = random_p_t()
    alg = QuantumSl2Quasi(p=p, t=t)
    checks = [
        ("check_all", axioms.check_all),
        ("check_r_matrix_intertwiner", axioms.check_r_matrix_intertwiner),
        ("check_hexagon", axioms.check_hexagon),
        ("check_ribbon", axioms.check_ribbon),
        ("check_s_delta_compatibility", axioms.check_s_delta_compatibility),
    ]
    return f"QuantumSl2Quasi(p={p}, t={t})", alg, checks


def _round_restricted_sl2():
    p = random_p()
    alg = RestrictedSl2(p=p)
    checks = [
        ("check_all", axioms.check_all),
        ("check_r_matrix_intertwiner", axioms.check_r_matrix_intertwiner),
        ("check_hexagon", axioms.check_hexagon),
        ("check_ribbon", axioms.check_ribbon),
        ("check_s_delta_compatibility", axioms.check_s_delta_compatibility),
        # generic drinfeld()/f_element() (QuasiHopfAlgebra) against the
        # classical-Hopf-algebra closed forms this trivial-Phi algebra
        # collapses to -- see tests/_special_elements.py.
        ("check_drinfeld_matches_classical_hopf_formula", _check_restricted_sl2_drinfeld),
        ("check_f_element_matches_trivial_formula", _check_restricted_sl2_f_element),
    ]
    return f"RestrictedSl2(p={p})", alg, checks


def _round_symplectic_fermion():
    N, beta_power = random_N_beta_power()
    alg = SymplecticFermionQ(N=N, beta_power=beta_power)
    checks = [
        ("check_all", axioms.check_all),
        ("check_r_matrix_intertwiner", axioms.check_r_matrix_intertwiner),
        ("check_hexagon", axioms.check_hexagon),
        ("check_ribbon", axioms.check_ribbon),
        ("check_s_delta_compatibility", axioms.check_s_delta_compatibility),
        # generic drinfeld()/f_element() (QuasiHopfAlgebra) against the
        # paper's own closed forms (eq:sqs+sqsinvbr-Q, eq:def:F-Q) --
        # see tests/_special_elements.py.
        ("check_drinfeld_matches_explicit_formula", _check_symplectic_fermion_drinfeld),
        ("check_f_element_matches_explicit_formula", _check_symplectic_fermion_f_element),
    ]
    return f"SymplecticFermionQ(N={N}, beta_power={beta_power})", alg, checks


ROUNDS = [_round_quantum_sl2_quasi, _round_restricted_sl2, _round_symplectic_fermion]


def _run_round(desc, alg, checks) -> bool:
    print(f"\n=== {desc} ===")
    for name, fn in checks:
        t0 = time.monotonic()
        ok = fn(alg, verbose=True)
        dt = time.monotonic() - t0
        print(f"  {name}: {'OK' if ok else 'FAILED'} ({dt:.1f}s)")
        if not ok:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minutes", type=float, default=60.0, help="time budget in minutes (default: 60)")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=12,
        help="axioms.SAMPLE_SIZE to use for this run (default: 12, vs axioms.py's own default of 6)",
    )
    args = parser.parse_args()

    axioms.SAMPLE_SIZE = args.sample_size
    print(f"soak_test: running for {args.minutes:.1f} minutes, axioms.SAMPLE_SIZE={axioms.SAMPLE_SIZE}")

    start = time.monotonic()
    deadline = start + args.minutes * 60
    round_counts = {r.__name__: 0 for r in ROUNDS}
    total_rounds = 0

    try:
        while time.monotonic() < deadline:
            make_round = ROUNDS[total_rounds % len(ROUNDS)]
            desc, alg, checks = make_round()
            ok = _run_round(desc, alg, checks)
            total_rounds += 1
            round_counts[make_round.__name__] += 1
            if not ok:
                elapsed = time.monotonic() - start
                print(f"\nsoak_test: FAILED after {total_rounds} model instances, {elapsed / 60:.1f} minutes.")
                print(f"  breakdown: {round_counts}")
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nsoak_test: interrupted.")

    elapsed = time.monotonic() - start
    print(f"\nsoak_test: completed {total_rounds} model instances over {elapsed / 60:.1f} minutes, no failures.")
    print(f"  breakdown: {round_counts}")


if __name__ == "__main__":
    main()
