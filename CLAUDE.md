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

The first worked example, `U_q^{(Phi)}sl(2)`
(`src/hopfsym/examples/quantum_sl2_quasi.py`), is a direct port of
`reference/hopf-Uqsl2-quasi.txt`, Ingo's own Mathematica code
implementing the algebra from Creutzig-Gainutdinov-Runkel,
arXiv:1712.07260 (Ingo is a co-author). When extending or debugging
that example, that Mathematica file is the ground truth to check
against, line by line if needed -- it's authoritative, not just a
rough reference.

A second example, `U_res sl(2)` with `K^p = 1`
(`src/hopfsym/examples/restricted_sl2.py`), is the honest Hopf algebra
of the paper's Section 4.1, but with `K` of order `p` rather than the
paper's `2p` (Ingo's request, confirmed against the reasoning in that
file's module docstring: `K^p` is already central of order <= 2 in
`QuantumSl2Quasi`, so this is a well-defined Hopf-ideal quotient, same
`q`, same field). Unlike the first example, there is *no* Mathematica
reference for this one -- it was derived directly from the paper and
cross-checked against `QuantumSl2Quasi` computationally (see
`tests/test_restricted_sl2.py`'s `QuotientConsistencyTests`, which
checks the quotient-homomorphism property exhaustively). If you extend
this example, that test (and the paper's Section 4.1) is what to check
against, not a Mathematica file.

Surprising fact worth knowing before touching either example again: the
paper explicitly says `\UresSL2` (`K^{2p} = 1`) has *no* R-matrix (end of
Sec. 4.1) -- but `RestrictedSl2` (`K^p = 1`) *does* have one, and from it
a monodromy/Drinfeld/ribbon element, all satisfying the same axioms as
`QuantumSl2Quasi`'s. This isn't a contradiction (the no-go is about the
bigger algebra `RestrictedSl2` is a quotient of, not about it), but it's
the kind of thing worth re-verifying rather than assuming if either
example's definition changes -- it was established computationally
(`tests/test_restricted_sl2.py`'s `BraidingTests`), not just asserted.
`RestrictedSl2.r_matrix()`/`.drinfeld()` have clean from-scratch closed
forms (see the module docstring's "Where these formulas come from");
`.ribbon()` does not (the K^p=1 quotient of the Gauss-sum exponent is
parity-in-p-dependent) and is instead computed by literally reusing
`QuantumSl2Quasi.ribbon()`'s formula with the final K-exponent reduced
mod p -- a legitimate, exact technique (not a numerical shortcut), just
worth knowing so it isn't mistaken for an oversight.

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

### A single algebra can need more than one cyclotomic field

This came up porting the R-matrix and cost real time to track down, so:
**a structure map can force a bigger field than the rest of the
algebra, even when the algebra itself only has one `p`.** In
`quantum_sl2_quasi.py`, `Delta`/`S`/`Phi` only ever need integer powers
of `q` and live in `Q(zeta_{2p})`, but the R-matrix formula contains
`q^(t^2/2)` -- and since the gauge parameter `t` is always odd, `t^2/2`
is a genuine half-integer exponent, not an artifact of how the formula
happens to be written. So the R-matrix (and everything built from it:
`monodromy`, `drinfeld`, hexagon, and -- via a different mechanism,
see below -- `ribbon`) lives one field up, in `Q(zeta_{4p})`. Before
assuming a new piece of structure lives in the same field as the rest
of an algebra, check every exponent of `q` it involves for a stray
`/2` (or similar) the way this was caught here -- don't assume the
existing field is big enough just because it was for everything ported
so far.

Two pieces make mixing coefficients from different (but nested)
cyclotomic fields transparent, so call sites never need to embed by
hand:

- `CycloNum.embed(new_n)`: the field embedding `Q(zeta_n) -> Q(zeta_{k*n})`
  via `zeta_n |-> zeta_{k*n}**k`, memoised per instance (this mattered:
  the same small-field coefficients -- e.g. `multiply_basis`'s cached
  results -- get embedded repeatedly once a big element like an
  R-matrix combines with them across hundreds of terms, so caching the
  embedding, not just the multiplication, was the fix that made
  `check_r_matrix_intertwiner` fast).
- `CycloNum` arithmetic (`__add__`/`__mul__`/`__truediv__`) auto-widens:
  combining `Q(zeta_{2p})` and `Q(zeta_{4p})` values just works (the
  smaller side is embedded automatically), and only genuinely
  incompatible degrees (neither divides the other) still raise. This is
  why `r_matrix()`, `Phi`, `comul(...)`, etc. can be freely multiplied
  together in `axioms.py` without any algebra-specific glue code.

The ribbon element needs an extra trick beyond this: its prefactor
`(1-i)/(2*sqrt(p))` isn't obviously an element of *any* cyclotomic field
as written (`i` and `sqrt(p)` look like they need the complex numbers).
The paper's derivation (`eq:gauss-sum`) rewrites it via the classical
quadratic Gauss sum `sum_{a=0}^{2p-1} q^(-a^2/2) == (1-i)*sqrt(p)`,
which *is* manifestly a finite sum of integer powers of `q^(1/2)` --
i.e. an honest, exactly-computable `CycloNum(4p)` element, no black-box
`sqrt`/complex arithmetic needed anywhere (see `ribbon()`'s docstring
for the one-line algebraic simplification that turns this into the
actual prefactor used). If a future algebra's structure constants
involve a similar-looking `sqrt`/`i` combination, look for the
corresponding Gauss-sum identity before reaching for a numerical
approximation -- it is very likely exactly representable the same way.

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
    associator_inv(), alpha(), beta() -- required; r_matrix(), ribbon(),
    drinfeld(), f_element() -- optional (quasi-triangular/ribbon
    structure only, default to NotImplementedError).

axioms.py
    generic checkers written only against that interface:
    check_associativity, check_bialgebra_homomorphism, check_counit,
    check_twisted_coassociativity, check_pentagon, check_antipode,
    check_evaluation_coevaluation, and check_all() to run the required
    ones together; check_r_matrix_intertwiner, check_hexagon,
    check_ribbon, check_s_delta_compatibility for the optional
    quasi-triangular/ribbon structure (not part of check_all, since not
    every algebra has it -- call these separately, see README).

examples/quantum_sl2_quasi.py
    the one concrete algebra so far. multiply_basis is implemented via
    _reduce_word, a small term-rewriting routine operating on a list of
    (letter, exponent) tokens -- this is *the* place the algebra's
    relations live; every other structure map (comul, antipode,
    r_matrix, ribbon, drinfeld, f_element, ...) only ever calls
    self.mul(), which bottoms out in _reduce_word.
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
6. If there's no ground-truth source to diff against line-by-line (as
   with `restricted_sl2.py`, which has no Mathematica reference), look
   for an *independent* cross-check instead of trusting internal
   self-consistency alone -- the axiom checks catch a lot, but they'd
   also happily pass on a self-consistent algebra that isn't the one you
   meant to define. `restricted_sl2.py` is verified as the exact
   quotient of `quantum_sl2_quasi.py` by `K^p = 1` (checked exhaustively
   over every basis pair, not sampled); `quantum_sl2_quasi.py`'s
   R-matrix/monodromy are similarly checked against independently
   re-derived closed forms (`testExpression` in the Mathematica source).
   A relationship like this between two examples is usually available
   whenever one is a variant/limit/quotient of another -- look for it.

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
- **Never accumulate into an `Element` via `result = result + piece`
  inside a loop.** `Element.__add__` copies the *entire* left-hand side
  (via `Element.__init__` re-inserting every existing term) on every
  call -- fine once, quadratic across a loop. This was `tensor_mul`'s
  bug (element.py): summing a ~1000-term result one small `piece` at a
  time this way made `check_r_matrix_intertwiner` take ~40s at `p=4`
  purely from re-copying its own growing result millions of times over
  -- nothing to do with the size of the actual computation. The fix was
  to accumulate directly via `result.add_term(k, c)` in the innermost
  loop instead (~26s just from that, before the embed-caching below).
  `algebra.py`'s `mul()`, `tensor()`, and `apply_to_factor()` already
  did it the right way; `tensor_mul` was the one holdout. If you add a
  new function that builds up an `Element` across many small pieces,
  use `add_term` directly, not repeated `+`.
- **Memoise field embeddings, not just field arithmetic.** Once one
  structure map lives in a bigger cyclotomic field than another (see
  "A single algebra can need more than one cyclotomic field" above),
  every mixed-field operation calls `CycloNum.embed()` -- and the same
  small-field coefficient objects (e.g. `multiply_basis`'s cache)
  recur across every pairing with a big element's hundreds of terms.
  `CycloNum.embed()` is memoised per instance for exactly this reason;
  dropping the cache measurably regressed `check_r_matrix_intertwiner`
  even after the `tensor_mul` fix above.

If you're chasing more speed later, `cProfile` +
`pstats.Stats(...).sort_stats("tottime")` on a single `axioms.check_*`
call is the fastest way to find where time is actually going -- and
`pstats.Stats(...).print_callers(...)` to find *who* is calling the hot
function when the raw call count doesn't match what the algorithm
should need (this is what caught the `tensor_mul` bug above: the
hot function was `Element.add_term`, but the real story was in its
caller, `Element.__init__`). Don't guess -- the bottleneck has moved
several times over this project's history (from `_comul_basis`'s
repeated squaring, to per-element exhaustive sampling, to `Fraction()`
construction, to `tensor_mul`'s accumulation pattern).

## Not yet ported (from the Mathematica reference)

Everything `testall` actually exercises is now ported: associativity,
`Delta` as an algebra homomorphism, the counit and antipode axioms,
twisted coassociativity, the pentagon identity, the
evaluation/coevaluation identities (`testEvalCoeval`), the R-matrix and
its intertwiner property (`testRinter`), the hexagon axioms
(`testHexagon`), the monodromy element, the Drinfeld element, the ribbon
element and its defining properties (`testRibbon`), the "F" element and
S-Delta compatibility (`testSDeltaR`), and the explicit closed-form
cross-checks (`testExpression`) -- see
`src/hopfsym/examples/quantum_sl2_quasi.py` (`r_matrix`, `monodromy`,
`drinfeld`, `ribbon`, `f_element`) and
`tests/test_quantum_sl2_quasi_braiding.py`.

Two pieces the *original Mathematica source itself* never finished are
correspondingly still absent here: the Hopf pairing (`hopfpair`) and the
monodromy-matrix non-degeneracy check (`testMonodromy`) are both left
commented out with a `TODO` in `reference/hopf-Uqsl2-quasi.txt` -- there
is no working reference to port them against yet.

If a new algebra needs the R-matrix/ribbon/Drinfeld machinery, the
pattern that worked here: implement `r_matrix()`/`ribbon()`/
`drinfeld()`/`f_element()` as methods on the algebra (they're optional in
`QuasiHopfAlgebra`, defaulting to `NotImplementedError`), and reuse the
generic checkers in `axioms.py`
(`check_r_matrix_intertwiner`/`check_hexagon`/`check_ribbon`/
`check_s_delta_compatibility`) rather than writing new ones.

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
