# %%
# Interactive usage examples for SymplecticFermionQ (Q(N, beta)) --
# same Jupyter-cell style as ../../../sandbox.py (VSCode: run cells with
# Shift+Enter via the Python extension's "Run Cell"), but scoped to just
# this one algebra. See symplectic_fermion.py's module docstring for the
# full definition.

import sys
from pathlib import Path


def _find_src_dir() -> Path:
    """Locate the project's src/ directory (containing the hopfsym
    package) by walking upward from wherever we can determine a
    starting point. __file__ is used when it's defined (running as a
    plain script, `python3 symplectic_fermion_usage.py`); the current
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
from hopfsym.element import Element, Δ, ε
from hopfsym.examples import SymplecticFermionQ

alg = SymplecticFermionQ(N=2, beta_power=0)  # beta = 1

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

# Generators. alg.K is a tagged property; alg.f(i, eps) (eps = '+' or
# '-', i 1-indexed matching the paper's f_i^+-) is a tagged method,
# since the generators are indexed rather than fixed-named like E/F/K.
# alg.e0/.e1 are the tagged central idempotents (1+-K^2)/2.
K = alg.K
f1p, f1m, f2p = alg.f(1, "+"), alg.f(1, "-"), alg.f(2, "+")

# %%
# The defining relations (see the module docstring), each written so it
# evaluates to exactly the zero Element -- a direct sanity check you can
# run after any change to _reduce_word. No alg.pretty() needed here:
# they're all zero, and Element.__repr__ (what print() falls back to)
# already special-cases the zero Element as exactly "0", same as
# pretty() would show -- the algebra-specific f/K formatting only
# matters for *non*-zero results, like the coproducts below.
print("{f_i^+-, K} = 0           :", f1p * K + K * f1p)
print("{f_i^+,f_j^-} - e1, i=j   :", f1p * f1m + f1m * f1p - alg.e1)
print("{f_i^eps,f_j^eps}, i!=j   :", f1p * f2p + f2p * f1p)
print("(f_i^eps)^2 = 0           :", f1p ** 2)
print("K^4 - 1 = 0               :", K ** 4 - alg.unit())

# %%
# Example coproducts. Delta(K) is *not* simply grouplike here (unlike
# QuantumSl2Quasi/RestrictedSl2's K): it has a correction term
# -(1+(-1)^N)(e1.K) (x) (e1.K) that's active for even N (N=2 here) and
# vanishes for odd N -- see _comul_K1's docstring. Delta(f_i^-) is the
# richer, genuinely twisted formula spelled out in the paper (see
# test_symplectic_fermion.py's PaperRemarkTests for the explicit
# (Delta x id)/(id x Delta) cross-check against it). Non-zero, so
# alg.pretty() is worth it here.
print(alg.pretty(Δ(K)))
print(alg.pretty(Δ(f1m)))

# %%
# epsilon(f_i^+-) = 0, epsilon(K) = 1.
print(ε(f1p), ε(f1m), ε(K))

# %%
# The associator -- trivial (1 (x) 1 (x) 1) here since N is even and
# beta = 1 (see the module docstring's Remark right after
# eq:Q-antipode-def); non-trivial in general.
print(alg.pretty(alg.associator()))

# %%
# The full required axiom set (associativity, Delta a homomorphism,
# counit, twisted coassociativity, pentagon, antipode,
# evaluation/coevaluation). check_all prints its own PASS/FAIL per axiom.
axioms.check_all(alg)

# %%
# The optional quasi-triangular/ribbon structure (R-matrix, hexagon,
# ribbon element) -- entirely within Q(zeta_8), no field extension
# needed (contrast QuantumSl2Quasi/RestrictedSl2). Unlike check_all,
# these only print internally on *failure*, so wrap in print() to see
# the (expected) True on success too.
print(axioms.check_r_matrix_intertwiner(alg))
print(axioms.check_hexagon(alg))
print(axioms.check_ribbon(alg))
print(alg.pretty(alg.ribbon()))

# %%
