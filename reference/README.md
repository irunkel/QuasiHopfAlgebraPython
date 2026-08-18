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
