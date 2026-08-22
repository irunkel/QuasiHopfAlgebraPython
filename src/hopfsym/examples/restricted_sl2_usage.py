# %%
# Interactive usage examples for RestrictedSl2 (U_res sl(2), K^p = 1) --
# same Jupyter-cell style as ../../../sandbox.py (VSCode: run cells with
# Shift+Enter via the Python extension's "Run Cell"), but scoped to just
# this one algebra. See restricted_sl2.py's module docstring for the
# full definition and how it relates to QuantumSl2Quasi.

import sys
from pathlib import Path


def _find_src_dir() -> Path:
    """Locate the project's src/ directory (containing the hopfsym
    package) by walking upward from wherever we can determine a
    starting point. __file__ is used when it's defined (running as a
    plain script, `python3 restricted_sl2_usage.py`); the current
    working directory is tried too, since VSCode's Interactive Window
    doesn't reliably set __file__ for cell-by-cell execution (it may be
    left undefined, or point at a copy/temp location) but does default
    cwd to the file's own directory."""
    starts = [Path.cwd()]
    try:
        starts.insert(0, Path(__file__).resolve().parent)
    except NameError:
        pass
    for start in starts:
        for candidate in (start, *start.parents):
            if (candidate / "src" / "hopfsym").is_dir():
                return candidate / "src"
    raise RuntimeError(
        "could not find the hopfsym src/ directory from __file__ or the "
        "current working directory -- run this from somewhere inside the "
        "QuasiHopfAlgebraPython checkout"
    )


sys.path.insert(0, str(_find_src_dir()))

from hopfsym import axioms
from hopfsym.element import Element, tensor, Δ, ε
from hopfsym.examples import RestrictedSl2

alg = RestrictedSl2(p=3)

# Make bare `elem` expressions auto-display via alg.pretty() in the
# Interactive Window -- see sandbox.py for why this hooks into IPython's
# display formatter rather than changing Element itself. No-ops when run
# as a plain script.
try:
    _ip = get_ipython()
except NameError:
    _ip = None
if _ip is not None:
    def _pretty_element(elem, p, cycle):
        p.text(alg.pretty(elem) if "alg" in globals() else repr(elem))

    _ip.display_formatter.formatters["text/plain"].for_type(Element, _pretty_element)

# Generators. alg.E/.F/.K are tagged properties (no parens) -- equivalent
# to alg.elt((1,0,0)) etc., just shorter to write; tagged so `*`/`**`
# mean alg.mul/repeated alg.mul directly.
E, F, K = alg.E, alg.F, alg.K

q, qinv = alg.q(1), alg.q(-1)
Kinv = alg.elt((0, 0, alg.p - 1))  # K^-1 = K^(p-1), since K^p = 1

# %%
# The defining relations (see the module docstring), each written so it
# evaluates to exactly the zero Element -- a direct sanity check you can
# run after any change to _reduce_word. No alg.pretty() needed here:
# they're all zero, and Element.__repr__ (what print() falls back to)
# already special-cases the zero Element as exactly "0", same as
# pretty() would show -- the algebra-specific E/F/K formatting only
# matters for *non*-zero results, like the coproducts below.
print("K E - q^2 E K            =", K * E - q ** 2 * E * K)
print("K F - q^-2 F K           =", K * F - q ** (-2) * F * K)
print("[E,F] - (K-K^-1)/(q-q^-1) =", E * F - F * E - (K - Kinv) * (1 / (q - qinv)))
print("E^p                      =", E ** alg.p)
print("F^p                      =", F ** alg.p)
print("K^p - 1                  =", K ** alg.p - alg.unit())

# %%
# Example coproducts -- the plain, untwisted formulas (no gauge
# parameter t, no central idempotents; this is an honest Hopf algebra,
# unlike QuantumSl2Quasi). Non-zero, so alg.pretty() is worth it here.
print(alg.pretty(Δ(K)))
print(alg.pretty(Δ(E)))
print(alg.pretty(Δ(F)))

# %%
# epsilon(E) = epsilon(F) = 0, epsilon(K) = 1.
print(ε(E), ε(F), ε(K))

# %%
# The associator is trivial here (alg.associator() == 1 (x) 1 (x) 1,
# alpha = beta = 1) -- unlike QuantumSl2Quasi, this really is an honest
# Hopf algebra.
print(alg.associator() == alg.associator_inv())

# %%
# The full required axiom set (associativity, Delta a homomorphism,
# counit, twisted coassociativity, pentagon, antipode,
# evaluation/coevaluation). check_all prints its own PASS/FAIL per axiom.
axioms.check_all(alg)

# %%
# The optional quasi-triangular/ribbon structure. Unlike QuantumSl2Quasi's
# namesake U_res sl(2) (K^{2p}=1, no R-matrix per the paper), this K^p=1
# quotient does have one -- see the module docstring for why. Unlike
# check_all, these only print internally on *failure*, so wrap in
# print() to see the (expected) True on success too.
print(axioms.check_r_matrix_intertwiner(alg))
print(axioms.check_hexagon(alg))
print(axioms.check_ribbon(alg))
print(alg.pretty(alg.ribbon()))

# %%
