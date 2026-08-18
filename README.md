# QuasiHopfAlgebraPython

A small, dependency-free Python framework (`hopfsym`) for symbolic
manipulation and axiom-checking of (quasi-)Hopf algebras.

The goal is not a general-purpose computer algebra system: each concrete
algebra hand-writes its own normal-form/rewriting rules (no
Groebner-basis machinery), but a shared, reusable core handles formal
linear combinations, tensor powers, and generic checks of the defining
axioms (associativity, coassociativity up to an associator, the
antipode axiom, the pentagon identity, ...). New algebras plug into
that shared core rather than reimplementing it.

## Status

The framework currently supports quasi-Hopf algebras (Hopf algebras are
the special case where the associator is trivial) over either a free
formal parameter `q` or `q` specialised to a root of unity -- whichever
a given algebra's relations require (see `src/hopfsym/qring.py`).

Worked example: `U_q^{(Phi)}sl(2)`, the quasi-Hopf algebra from
Creutzig-Gainutdinov-Runkel, ["A quasi-Hopf algebra for the triplet
vertex operator algebra"](https://arxiv.org/abs/1712.07260), ported
from the original Mathematica implementation
(`reference/hopf-Uqsl2-quasi.txt`). Verified symbolically (exactly, not
numerically) for a range of `(p, t)`: associativity, coassociativity up
to the associator (the "pentagon-adjacent" axiom), `Delta` being an
algebra homomorphism, the counit axioms, the pentagon identity, and the
(quasi-)antipode axiom.

Not yet ported: the R-matrix, ribbon element, hexagon axioms, and
Drinfeld element. See `CLAUDE.md` for the plan and for the
architecture in more depth.

## Quick example

```python
from hopfsym.examples import QuantumSl2Quasi
from hopfsym import axioms

alg = QuantumSl2Quasi(p=3, t=1)   # 2p'th root of unity, dimension 2p^3 = 54
axioms.check_all(alg)             # runs every axiom check, prints PASS/FAIL
```

## Running the tests

No external dependencies (not even pytest) -- everything uses the
standard library, so:

```bash
python3 -m unittest discover -s tests -v
```

## Layout

```
src/hopfsym/
    element.py        # Element (formal linear combos) + tensor bookkeeping
    algebra.py         # the QuasiHopfAlgebra interface
    axioms.py          # generic axiom checkers, written against that interface
    qring.py           # two coefficient rings: free q, and q a root of unity
    examples/
        quantum_sl2_quasi.py   # U_q^{(Phi)}sl(2), see above
tests/
reference/
    hopf-Uqsl2-quasi.txt        # original Mathematica source this was ported from
```

See `CLAUDE.md` for the architecture write-up, the math background, and
notes for adding a new algebra.
