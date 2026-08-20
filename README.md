# QuasiHopfAlgebraPython

A small, dependency-free Python framework (`hopfsym`) for symbolic
manipulation and axiom-checking of (quasi-)Hopf algebras.

The goal is not a general-purpose computer algebra system: each concrete
algebra hand-writes its own normal-form/rewriting rules (no
Groebner-basis machinery), but a shared, reusable core handles formal
linear combinations, tensor powers, and generic checks of the defining
axioms. New algebras plug into that shared core rather than
reimplementing it.

## Status

Supports quasi-Hopf algebras (Hopf algebras are the special case where
the associator is trivial) over either a free formal parameter `q` or
`q` specialised to a root of unity, whichever an algebra's relations
require (see `src/hopfsym/qring.py`). Three worked examples, each
verified exactly (not numerically) against `axioms.py`'s full axiom set:

- **`QuantumSl2Quasi`** -- `U_q^{(Phi)}sl(2)` from Creutzig-Gainutdinov-Runkel,
  ["A quasi-Hopf algebra for the triplet vertex operator
  algebra"](https://arxiv.org/abs/1712.07260), including the R-matrix,
  hexagon axioms, and the monodromy/Drinfeld/ribbon elements.
- **`RestrictedSl2`** -- `U_res sl(2)` with `K^p = 1` (the paper's honest
  Hopf algebra has `K^{2p} = 1`). Verified as the exact quotient of
  `QuantumSl2Quasi` by `K^p = 1`. Surprisingly, unlike its `K^{2p}=1`
  namesake (which the paper shows has *no* R-matrix), this quotient
  *does* have one.
- **`SymplecticFermionQ`** -- `Q(N, beta)`, the symplectic fermion ribbon
  quasi-Hopf algebra from Farsad-Gainutdinov-Runkel, ["The symplectic
  fermion ribbon quasi-Hopf algebra and the SL(2,Z)-action on its
  centre"](https://arxiv.org/abs/1706.08164). Fermionic generators, its
  own normal-form reduction; only the quasi-Hopf data is ported so far
  (no R-matrix/ribbon element yet).

See each example's module docstring (`src/hopfsym/examples/`) for its
precise definition and derivations, and `CLAUDE.md` for the architecture.

## Quick example

```python
from hopfsym.examples import QuantumSl2Quasi
from hopfsym import axioms

alg = QuantumSl2Quasi(p=3, t=1)   # 2p'th root of unity, dimension 2p^3 = 54
axioms.check_all(alg)             # runs the core quasi-Hopf axioms, prints PASS/FAIL

# The R-matrix/ribbon structure is optional (not every algebra has one),
# so it's checked separately:
axioms.check_r_matrix_intertwiner(alg)
axioms.check_hexagon(alg)
axioms.check_ribbon(alg)
```

Each example also has a runnable `..._usage.py` walkthrough (defining
relations, example coproducts, then the axiom checks) in
`src/hopfsym/examples/` -- run as a plain script, or step through
cell-by-cell in an editor that understands `# %%` markers (e.g. VSCode's
Python extension, see below).

## Running the interactive examples in VSCode

The `..._usage.py` files are meant to be stepped through cell by cell
(each `# %%` marker starts a new cell) rather than run all at once, so
you can inspect each result before moving to the next. VSCode supports
this via two free extensions. Starting from nothing:

1. **Install VSCode.** Download it from
   [code.visualstudio.com](https://code.visualstudio.com/) and run the
   installer (Windows/Mac/Linux all supported).
2. **Install the Python extension.** In VSCode, open the Extensions
   view (the icon in the left sidebar that looks like four squares, or
   `Ctrl+Shift+X` / `Cmd+Shift+X`), search for "Python", and install the
   one published by Microsoft. Search for "Jupyter" too and install
   that one as well if it isn't pulled in automatically -- it's what
   gives `# %%` cells their "Run Cell" buttons.
3. **Clone the repository.** You need `git` (comes with most systems
   already, otherwise get it from [git-scm.com](https://git-scm.com/)).
   Open a terminal, `cd` into whichever folder you want the project to
   live in, and run:

   ```bash
   git clone https://github.com/irunkel/QuasiHopfAlgebraPython.git
   ```

   This creates a `QuasiHopfAlgebraPython` subfolder with the full
   project in it.
4. **Open the project in VSCode.** Either run `cd QuasiHopfAlgebraPython`
   then `code .` in that same terminal, or, from VSCode's File menu, use
   "Open Folder..." and pick the `QuasiHopfAlgebraPython` folder you
   just cloned.
5. **Open a usage file**, e.g.
   `src/hopfsym/examples/restricted_sl2_usage.py`, from the Explorer
   sidebar on the left.
6. **Run the first cell.** Click "Run Cell" above the first `# %%`
   block (or put the cursor in it and press `Shift+Enter`). The first
   time, VSCode asks which Python interpreter to use -- any Python 3
   works, since the project has no external dependencies to install.
   This opens an "Interactive Window" panel showing that cell's output.
7. **Keep going.** `Shift+Enter` runs the current cell and moves to the
   next, so you can step through the defining relations, coproducts and
   axiom checks one at a time -- or use "Run All"/"Run All Cells" at the
   top of the file to run the whole thing in one go.

## Running the tests

No external dependencies (not even pytest) -- everything uses the
standard library.

```bash
python3 -m unittest discover -s tests -v   # the regular suite, ~1 minute
python3 tests/soak_test.py                 # long-running stress test (default: 1 hour)
```

`soak_test.py` cycles through fresh random instances of all three
examples for a configurable time budget (`--minutes`), covering far more
of the parameter space than a single run of the regular suite -- useful
after a change to shared machinery (`element.py`/`algebra.py`/
`axioms.py`) that every example depends on. See its module docstring for
options; stops at the first failure with the failing model's exact
constructor arguments printed for reproducibility.

## Layout

```
src/hopfsym/
    element.py          # Element (formal linear combos) + tensor bookkeeping
    algebra.py          # the QuasiHopfAlgebra interface
    axioms.py           # generic axiom checkers, written against that interface
    qring.py            # two coefficient rings: free q, and q a root of unity
    examples/
        quantum_sl2_quasi.py(_usage.py)    # U_q^{(Phi)}sl(2), see above
        restricted_sl2.py(_usage.py)       # U_res sl(2) with K^p = 1, see above
        symplectic_fermion.py(_usage.py)   # Q(N, beta), see above
tests/
    soak_test.py        # long-running stress test, see above
```

See `CLAUDE.md` for the architecture write-up, the math background, and
notes for adding a new algebra.
