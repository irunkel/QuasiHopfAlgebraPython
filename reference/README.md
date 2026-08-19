# Reference material

`hopf-Uqsl2-quasi.txt` is Ingo's original Mathematica implementation of
the quasi-Hopf algebra `U_q^{(Phi)}sl(2)`, kept here as the ground truth
that `src/hopfsym/examples/quantum_sl2_quasi.py` is a direct port of.
When debugging that example, diff the Python logic against this file
line by line rather than re-deriving from the paper -- it already
encodes all the correct formulas.

The source paper is Creutzig, Gainutdinov, Runkel, "A quasi-Hopf
algebra for the triplet vertex operator algebra",
[arXiv:1712.07260](https://arxiv.org/abs/1712.07260) (not copied into
this repo -- fetch it directly if needed).

`src/hopfsym/examples/symplectic_fermion.py` (`SymplecticFermionQ`,
Q(N, beta)) has no Mathematica reference -- it was transcribed directly
from Section 3.1 ("Definition of Q") of Farsad, Gainutdinov, Runkel,
"The symplectic fermion ribbon quasi-Hopf algebra and the
SL(2,Z)-action on its centre",
[arXiv:1706.08164](https://arxiv.org/abs/1706.08164) (also not copied
into this repo; the full `.tex` source is large and covers much more
than this one algebra -- fetch it directly, or see Ingo's local
`~/ClaudeFolder/HopfAlgebraTesting/arXiv-1706.08164v3/` checkout, if
re-deriving something). See that module's docstring for how its
correctness was verified in the absence of a line-by-line reference
(the paper's own Remark right after eq:Q-antipode-def gives two
independent, directly checkable facts, used as regression tests).
