# CLAUDE.md

Project-specific notes for whichever Claude session (Cowork or Claude
Code CLI) is working on this repo next. Read this before making
architectural changes -- the design choices below were made
deliberately (several after debugging real mistakes), not arbitrarily.

## What this project is

Ingo's long-running (months/years) personal tool for symbolic
computation with (quasi-)Hopf algebras, starting from research need:
verifying, by exact symbolic computation, the defining axioms of
specific quasi-Hopf algebras that come up in his research (vertex
operator algebras / logarithmic CFT / quantum groups). It's explicitly
*not* meant to become a general computer-algebra system -- keep additions
light and readable over clever/general.

The first (and so far only) worked example, `U_q^{(Phi)}sl(2)`
(`src/hopfsym/examples/quantum_sl2_quasi.py`), is a direct port of
`reference/hopf-Uqsl2-quasi.txt`, Ingo's own Mathematica code
implementing the algebra from Creutzig-Gainutdinov-Runkel,
arXiv:1712.07260 (Ingo is a co-author). When extending or debugging
that example, that Mathematica file is the ground truth to check
against, line by line if needed -- it's authoritative, not just a
rough reference.

## Design philosophy (please preserve these choices)

- **No Groebner-basis / automatic rewriting engine.** Each concrete
  algebra hand-writes its own normal-form reduction (see
  `QuantumSl2Quasi._reduce_word`). This was an explicit choice, not a
  shortcut -- keep it that way for new examples too.
- **A single interface, not a class hierarchy.** There's no
  `HopfAlgebra extends Bialgebra extends Algebra` chain. Every concrete
  algebra implements `QuasiHopfAlgebra` (`src/hopfsym/algebra.py`); an
  honest Hopf algebra is just the special case where `associator()` is
  the trivial element and `alpha()`/`beta()` are both the unit. This
  keeps `axioms.py` written once, generically, and it already covers
  both cases.
- **Exact symbolic arithmetic, no black-box `simplify()`.** Both
  coefficient rings in `qring.py` do exact arithmetic (rational
  functions, or reduction modulo an exact cyclotomic polynomial), so
  equality-with-zero is always a real computation, never an
  approximation or a heuristic.

## Generic q vs root-of-unity q -- read this before adding an example

This is the single most important mathematical gotcha in this codebase,
and it cost real debugging time to discover (see git history / the
`quantum_sl2_quasi.py` docstring for the story), so: **check this before
writing a new algebra's structure constants.**

`src/hopfsym/qring.py` provides two coefficient rings:

- `QRational` (+ `LaurentPoly`): `Q(q)`, q a completely free formal
  variable. Use this when the algebra's relations hold for *any* q.
- `CycloNum` (+ `Poly`, `cyclotomic_polynomial`): `Q(zeta_n)`, q
  specialised to an actual primitive n'th root of unity. Use this when
  the relations only make sense for such a q.

How to tell which one a new algebra needs: look for a relation that
both (a) puts a generator to a fixed finite order (like `K^{2p} = 1`)
and (b) has that same generator act on another generator by
conjugation with a q-dependent scalar (like `K E K^-1 = q^2 E`).
Conjugating (b) around the cycle implied by (a) forces a relation on q
itself (`q^{4p} = 1` in that example) -- if that happens, q is *not*
free, and you need `CycloNum` with the right `n`. If no such relation
exists, `QRational` is simpler and faster and should be preferred.
Don't assume; re-derive the necessary condition for each new algebra
the way `qring.py`'s docstring does for this one.

Also note: even after pinning down that q must satisfy `q^n = 1` for
some n, you may separately need `q - q^-1` (or similar) to be
*invertible*, which requires working in the actual field `Q(zeta_n)`
(quotient by the irreducible cyclotomic polynomial) rather than the
easier-but-not-a-field `Q[x]/(x^n - 1)` (which has zero divisors
whenever n is composite or even). `CycloNum` already does this
correctly; just make sure you pick the right `n`.

`Element` (`element.py`) itself is fully agnostic to which ring is in
use -- it never imports `QRational` or `CycloNum` by name, it just
calls `+`, `-`, `*`, `.is_zero()` on whatever coefficient objects it's
given (plain `int`/`Fraction` mix in fine too, via each ring's own
`__radd__`/`__rmul__`). A new ring type just needs to support that same
small surface.

## Architecture

```
Element (element.py)
    a dict {basis_key: coefficient}. Doesn't know what a basis key
    means -- that's up to the algebra. TensorKey marks a basis key as
    an n-fold tensor of other keys (tensor(), apply_to_factor(),
    tensor_mul() build/manipulate these generically).

QuasiHopfAlgebra (algebra.py)
    the interface: basis(), unit(), multiply_basis() [+ mul(), given
    for free], comul(), counit(), antipode(), associator(),
    associator_inv(), alpha(), beta().

axioms.py
    generic checkers written only against that interface:
    check_associativity, check_bialgebra_homomorphism, check_counit,
    check_twisted_coassociativity, check_pentagon, check_antipode,
    and check_all() to run everything.

examples/quantum_sl2_quasi.py
    the one concrete algebra so far. multiply_basis is implemented via
    _reduce_word, a small term-rewriting routine operating on a list of
    (letter, exponent) tokens -- this is *the* place the algebra's
    relations live; every other structure map (comul, antipode, ...)
    only ever calls self.mul(), which bottoms out in _reduce_word.
```

## Adding a new algebra

1. Work out its presentation (generators, relations, a normal-form
   basis) and its structure maps, on paper or from a reference.
2. Decide generic-q vs root-of-unity q (see above) and derive the right
   `n` if it's the latter.
3. Subclass `QuasiHopfAlgebra`, implement `multiply_basis` first (get
   `check_associativity` passing before anything else -- it's the
   cheapest check and the one most likely to catch a normal-form bug),
   then `comul`/`counit`, then `antipode`/`associator`/`alpha`/`beta`
   if applicable.
4. Add a test file under `tests/` mirroring
   `tests/test_quantum_sl2_quasi.py`: run `axioms.check_all` (or the
   individual checks) for a few small parameter choices.
5. If you hand-derive a formula from a paper or existing code (as with
   the Mathematica port here), keep the source close by (a `reference/`
   file, or a docstring citation) -- when something fails, being able
   to diff against ground truth line-by-line is what actually finds the
   bug, as opposed to re-deriving from scratch under time pressure.

## Performance notes

Exact symbolic computation over a cyclotomic field is not free, and
speed was a real pain point in the original Mathematica code too --
worth actively watching here, not just an incidental concern.

What's already in place:

- **Memoize everything that only depends on small integer indices.**
  `QuantumSl2Quasi` caches `multiply_basis` results, `comul`/`antipode`
  results per basis triple, and -- the change with the biggest
  effect -- the coproduct/antipode of each *individual generator power*
  (`Delta(E)^a` for each `a` separately, not recomputed from scratch for
  every basis element that happens to have that `a`). Before this,
  `check_twisted_coassociativity` for `p=4` took ~40s; after, ~6s (and
  `p=5`, `p=6` went from "doesn't finish in 2 minutes" to ~25s/~70s).
  If you add a new structure map with a similar "power of a generator"
  pattern, cache the power itself, not just the final per-basis-element
  result.
- **Sample by default, don't be exhaustive.** `axioms.py`'s per-element
  checks (counit, twisted coassociativity, antipode) test a random
  sample of the basis by default, not every element -- the same
  approach the original Mathematica code took (its `ntest`-many random
  tests), for the same reason: the coproduct of a single basis element
  can expand into dozens of terms, and checking all of them for every
  one of `2p^3` basis elements gets expensive fast. Pass
  `samples=alg.basis()` explicitly when you want an exhaustive check
  (cheap for small `p`, e.g. `p=2`, dimension 16).
- `Poly.__init__` (in `qring.py`) skips redundant `Fraction()`
  conversion when a coefficient is already a `Fraction` -- this showed
  up as the single hottest line in profiling before the fix.

If you're chasing more speed later, `cProfile` +
`pstats.Stats(...).sort_stats("tottime")` on a single `axioms.check_*`
call is the fastest way to find where time is actually going -- don't
guess, the bottleneck moved twice already during this project's first
session (from `_comul_basis`'s repeated squaring, to per-element
exhaustive sampling, to `Fraction()` construction).

## Not yet ported (from the Mathematica reference)

- The universal R-matrix and hexagon axioms (quasi-triangularity).
- The ribbon element and its defining properties.
- The Drinfeld element, and the "F" element relating `Delta` and `S`
  (`testSDeltaR` in the Mathematica source).
- The explicit closed-form cross-checks (`testExpression` in the
  Mathematica source).

These all build on the core that's now in place; porting them should
follow the same pattern (hand-derive/port the formula into a method on
`QuantumSl2Quasi`, add a generic checker to `axioms.py` if the property
is general enough to be reusable, add a regression test).

## Planned generalization: quasi-Hopf group-coalgebras

Ingo's longer-term direction after this: quasi-Hopf **group-coalgebras**
(Turaev), where instead of one algebra `H` with `Delta: H -> H (x) H`,
you have a family `{H_alpha}` indexed by a group `G`, with
`Delta_{alpha,beta}: H_{alpha*beta} -> H_alpha (x) H_beta`, and
associators/antipodes indexed by group elements too. This is a real
generalization, not just a bigger example -- don't try to preemptively
build the group-indexed machinery now (that would violate the "keep it
light" principle above before it's needed), but when it comes up:

- The natural seam is that every structure map in `QuasiHopfAlgebra`
  implicitly operates "at the identity element of a trivial group" --
  `comul`, `associator`, etc. could grow an optional group-element
  parameter defaulting to the identity, recovering exactly today's
  behaviour when unused.
- `axioms.py`'s checkers would need the analogous group-indexed
  versions of each identity (the group-coalgebra pentagon/antipode
  axioms involve specific triples/pairs of group elements rather than
  always the same map).
- Don't design this speculatively; wait until there's a first concrete
  group-coalgebra example to port, the same way `QuasiHopfAlgebra`'s
  current shape was driven by actually porting `quantum_sl2_quasi.py`
  rather than guessed in advance.

## Where this project lives

- Local checkout on Ingo's machine: `~/ClaudeFolder/QuasiHopfAlgebraPython`
  -- a real git clone with SSH push access to
  https://github.com/irunkel/QuasiHopfAlgebraPython. This is the
  primary place for day-to-day development now.
- Cloud sandbox (Cowork sessions): `~/QuasiHopfAlgebraPython` (or
  wherever the session's working directory puts it). Used for longer
  exploratory sessions. It has no direct GitHub access, so any changes
  made there need a git bundle to transfer back to the local checkout.
- GitHub: https://github.com/irunkel/QuasiHopfAlgebraPython (the
  canonical source of truth / durable history across sessions).
- The original Mathematica reference material (the arXiv PDF and the
  `.tex` source) lives in `~/ClaudeFolder/HopfAlgebraTesting` on Ingo's
  machine; only the Mathematica `.txt` source is copied into this
  repo's `reference/` folder, since it's what the port is checked
  against directly.

## Running tests

```bash
python3 -m unittest discover -s tests -v
```

No dependencies beyond the standard library -- this is deliberate (see
`qring.py`'s docstring): the original plan to depend on sympy hit a
network-egress restriction in the Cowork sandbox mid-project, and the
dependency-free rewrite turned out better anyway (exact arithmetic, no
`simplify()` black box, works instantly in any environment including
inside Claude Code CLI with zero setup).
