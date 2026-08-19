"""Formal linear combinations, and tensor-power bookkeeping, that are
shared by every algebra in this package.

An :class:`Element` is just ``sum_i coeff_i * basis_i``: a dict from
*basis keys* to coefficients. This module has no idea what a basis key
*means* -- that is entirely up to whichever
:class:`~hopfsym.algebra.QuasiHopfAlgebra` created the Element -- and
it is equally agnostic about what ring the coefficients live in: a
coefficient can be a plain ``int``/``Fraction``, a
:class:`hopfsym.qring.QRational` (free q), a
:class:`hopfsym.qring.CycloNum` (q a root of unity), or in principle
anything else that supports ``+``, unary/binary ``-``, ``*``, and
``.is_zero()`` -- see :mod:`hopfsym.qring` for why an algebra might need
one ring or the other. All this module provides is the additive
vector-space structure over whichever ring is in play, plus generic
n-fold tensor-product bookkeeping (building ``H (x) H``,
``H (x) H (x) H``, ... elements, and applying a structure map to one
factor of such a tensor -- used for axioms like the pentagon identity,
which apply the coproduct to one leg of the associator).
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple


def _is_zero(coeff) -> bool:
    is_zero = getattr(coeff, "is_zero", None)
    return is_zero() if is_zero is not None else coeff == 0


class TensorKey(tuple):
    """A basis key for an n-fold tensor product: a flat tuple of the
    individual factors' own basis keys.

    This is deliberately a distinct type from a plain tuple (even though
    it behaves like one) so that ``tensor()`` can tell "one basis key
    that happens to be a tuple" apart from "several basis keys tensored
    together", regardless of what an individual algebra uses as its own
    basis key representation.
    """

    def __new__(cls, factors):
        return super().__new__(cls, factors)

    def __repr__(self):
        return " (x) ".join(repr(f) for f in self)


def _factors_of(key) -> tuple:
    """View any basis key as a tuple of tensor factors: a TensorKey is
    already such a tuple; anything else counts as a single factor."""
    return tuple(key) if isinstance(key, TensorKey) else (key,)


def _key_from_factors(factors: tuple):
    """Inverse of _factors_of: a 1-tuple collapses back to a plain key,
    anything longer becomes a TensorKey."""
    return factors[0] if len(factors) == 1 else TensorKey(factors)


def _arity(elem: "Element") -> int:
    """1 for a plain (single-factor) Element, n for an n-fold tensor
    Element (basis keys are length-n TensorKeys); 1 for the zero element
    itself (arity is moot there -- dispatching either way below gives
    the correct empty result)."""
    if not elem.terms:
        return 1
    key = next(iter(elem.terms))
    return len(key) if isinstance(key, TensorKey) else 1


def _combine_alg(a, b):
    """Resolve the ``.alg`` tag of a result from two operands' tags.
    ``None`` acts as a wildcard (an untagged Element, e.g. anything
    built via ``tensor()``/``tensor_mul()``/``apply_to_factor()``, or a
    plain ``Element.basis(key)`` call, combines fine with a tagged one --
    the same "None is compatible with anything" convention
    ``qring.CycloNum`` uses for its field degree). Two *different*
    concrete tags is almost certainly a bug (mixing Elements from two
    different algebra instances), so that raises rather than silently
    picking one."""
    if a is None:
        return b
    if b is None or b is a:
        return a
    raise ValueError(
        "combining Elements tagged with two different algebra instances -- "
        "this usually means basis elements from two different algebra() "
        "objects got mixed together"
    )


class Element:
    """A formal linear combination ``{basis_key: coefficient}``.

    ``alg``, if set, tags an Element as belonging to a specific
    :class:`~hopfsym.algebra.QuasiHopfAlgebra` instance -- purely so that
    ``*`` (and ``**``, repeated ``*``) between two such Elements can mean
    that algebra's product instead of raising (see ``__mul__``/
    ``__pow__``): ``alg.mul`` for two plain (single-factor) Elements, or
    the componentwise tensor-algebra product (``tensor_mul``,
    ``(a(x)b)*(c(x)d) = (a*c)(x)(b*d)``) when either side is an n-fold
    tensor Element -- which arity applies is read off the operands'
    basis keys, not tracked separately. It is optional and propagated
    automatically through ``+``/``-``/scalar ``*``/``tensor()``; nothing
    else in this module ever reads or requires it, so untagged use
    (``alg=None`` everywhere, the default) behaves exactly as before
    this was added. Get a tagged Element via ``QuasiHopfAlgebra.elt()``
    (or a per-example generator accessor built on it, e.g.
    ``RestrictedSl2.E``) or ``Element.basis(key, alg=...)``, or
    implicitly from ``alg.mul(...)``/``tensor(...)``'s result when the
    inputs were tagged.
    """

    __slots__ = ("terms", "alg")

    def __init__(self, terms: Dict = None):
        self.terms: Dict = {}
        self.alg = None
        if terms:
            for k, v in terms.items():
                self.add_term(k, v)

    @classmethod
    def zero(cls) -> "Element":
        return cls()

    @classmethod
    def basis(cls, key, coeff=1, alg=None) -> "Element":
        e = cls({key: coeff})
        e.alg = alg
        return e

    def add_term(self, key, coeff) -> None:
        if _is_zero(coeff):
            return
        if key in self.terms:
            new = self.terms[key] + coeff
            if _is_zero(new):
                del self.terms[key]
            else:
                self.terms[key] = new
        else:
            self.terms[key] = coeff

    def __add__(self, other: "Element") -> "Element":
        if not isinstance(other, Element):
            return NotImplemented
        result = Element(dict(self.terms))
        for k, v in other.terms.items():
            result.add_term(k, v)
        result.alg = _combine_alg(self.alg, other.alg)
        return result

    def __neg__(self) -> "Element":
        result = Element({k: -v for k, v in self.terms.items()})
        result.alg = self.alg
        return result

    def __sub__(self, other: "Element") -> "Element":
        if not isinstance(other, Element):
            return NotImplemented
        return self + (-other)

    def __rmul__(self, scalar) -> "Element":
        result = Element({k: scalar * v for k, v in self.terms.items()})
        result.alg = self.alg
        return result

    def __mul__(self, other) -> "Element":
        if isinstance(other, Element):
            resolved = _combine_alg(self.alg, other.alg)
            if resolved is not None:
                if _arity(self) > 1 or _arity(other) > 1:
                    # One or both operands are n-fold tensor elements
                    # (e.g. built via tensor(a, b)): `*` means the
                    # componentwise tensor-algebra product,
                    # (a(x)b)*(c(x)d) = (a*c)(x)(b*d), i.e. tensor_mul --
                    # not multiply_basis, which only knows plain,
                    # single-factor basis keys. tensor_mul itself raises
                    # if the two arities don't actually match.
                    result = tensor_mul(resolved, self, other)
                    result.alg = resolved
                    return result
                return resolved.mul(self, other)
            raise TypeError(
                "Element * Element is ambiguous (algebra product or tensor?); "
                "use alg.mul(a, b) or tensor(a, b) explicitly -- or tag at "
                "least one operand with its algebra (e.g. via "
                "QuasiHopfAlgebra.elt(key)) to make `*` mean the algebra product"
            )
        return self.__rmul__(other)

    def __pow__(self, n: int) -> "Element":
        """Repeated algebra product, self * self * ... * self (n times);
        requires a tagged Element for the same reason ``*`` does (see
        __mul__) -- there is no algebra to multiply in otherwise. n=0
        gives the algebra's unit. Square-and-multiply, same pattern as
        qring.CycloNum.__pow__."""
        if not isinstance(n, int):
            return NotImplemented
        if n < 0:
            raise ValueError("Element ** n needs n >= 0 (no general inverses)")
        if self.alg is None:
            raise TypeError(
                "Element ** n requires a tagged Element (e.g. via alg.elt(key) "
                "or alg.E/alg.F/alg.K) so `**` knows which algebra's product to use"
            )
        alg = self.alg
        # The multiplicative identity of matching tensor arity: alg.unit()
        # itself for a plain (arity-1) Element, 1(x)1(x)...(x)1 (arity
        # copies) for an n-fold tensor Element -- so n=0 and the
        # square-and-multiply loop below both stay correct at any arity.
        result = alg.unit()
        for _ in range(_arity(self) - 1):
            result = tensor(result, alg.unit())
        result.alg = alg
        base = self
        while n > 0:
            if n & 1:
                result = result * base
            n >>= 1
            if n:
                base = base * base
        return result

    def is_zero(self) -> bool:
        return len(self.terms) == 0

    def __eq__(self, other):
        if not isinstance(other, Element):
            return NotImplemented
        return (self - other).is_zero()

    def __repr__(self):
        if not self.terms:
            return "0"
        return " + ".join(f"({v})*{k!r}" for k, v in self.terms.items())


def tensor(a: Element, b: Element) -> Element:
    """The bilinear tensor product of two elements. Repeated application
    (e.g. ``tensor(tensor(a, b), c)`` or ``tensor(a, tensor(b, c))``)
    flattens automatically into one n-fold TensorKey, regardless of how
    the calls are nested.

    Propagates ``.alg`` the same way ``+``/``-`` do (``_combine_alg``,
    ``None`` as wildcard): tensoring two elements tagged with the same
    algebra gives a tagged n-fold tensor Element, so e.g.
    ``tensor(a, b) * tensor(c, d)`` can mean ``tensor(a*c, b*d)`` (see
    ``Element.__mul__``) rather than raising. Tensoring an untagged
    piece (the common case -- most internals use plain
    ``Element.basis(key)``) still yields an untagged result, exactly as
    before."""
    result = Element()
    for k1, c1 in a.terms.items():
        for k2, c2 in b.terms.items():
            key = _key_from_factors(_factors_of(k1) + _factors_of(k2))
            result.add_term(key, c1 * c2)
    result.alg = _combine_alg(a.alg, b.alg)
    return result


def permute_factors(elem: Element, perm: Tuple[int, ...]) -> Element:
    """Reorder the tensor factors of an n-fold tensor element: the factor
    at position ``i`` of the result is the factor at position ``perm[i]``
    of ``elem`` (so ``perm`` is a tuple of the same length as the tensor
    arity, e.g. ``(1, 2, 0)`` sends ``tens[a1,a2,a3]`` to
    ``tens[a2,a3,a1]``). Used for axioms that need a specific leg
    permutation of the associator, e.g. the hexagon axiom, and generalises
    the original Mathematica code's ``flip`` (the 2-factor case,
    ``perm=(1,0)``)). Propagates ``.alg`` from ``elem`` (same convention
    as ``tensor()``), since a permutation doesn't change which algebra
    the factors belong to."""
    result = Element()
    for key, coeff in elem.terms.items():
        factors = _factors_of(key)
        new_factors = tuple(factors[i] for i in perm)
        result.add_term(_key_from_factors(new_factors), coeff)
    result.alg = elem.alg
    return result


def flip(elem: Element) -> Element:
    """Swap the two factors of a 2-fold tensor element."""
    return permute_factors(elem, (1, 0))


def apply_to_factor(elem: Element, position: int, func: Callable[[Element], Element]) -> Element:
    """Apply the linear map ``func`` to the tensor factor at ``position``
    (0-indexed) of every term of ``elem``, leaving the other factors
    untouched. If ``func`` itself returns a multi-factor tensor element
    (e.g. a coproduct), the arity grows accordingly.

    This is the generic building block behind axiom checks that need
    things like ``(Delta (x) id)(x)`` or ``(id (x) id (x) Delta)(x)``.

    Propagates ``.alg`` from ``elem`` to both the single-factor piece
    handed to ``func`` and the overall result (same convention as
    ``tensor()``), so ``func`` can itself be a tag-dispatching function
    like ``Δ`` (e.g. ``apply_to_factor(da, 0, Δ)`` for ``(Δ (x) id)(da)``)
    when ``elem`` is tagged; an untagged ``elem`` behaves exactly as
    before (``func`` receives an untagged piece, as it always did)."""
    result = Element()
    for key, coeff in elem.terms.items():
        factors = _factors_of(key)
        before, target, after = factors[:position], factors[position], factors[position + 1:]
        mapped = func(Element.basis(target, coeff, alg=elem.alg))
        for mkey, mcoeff in mapped.terms.items():
            mfactors = _factors_of(mkey)
            newfactors = before + mfactors + after
            result.add_term(_key_from_factors(newfactors), mcoeff)
    result.alg = elem.alg
    return result


def tensor_mul(alg, a: Element, b: Element) -> Element:
    """Multiply two elements of ``H^{(x)n}`` componentwise, using
    ``alg.mul`` on each pair of matching factors -- i.e. the algebra
    structure on the n-fold tensor power. Works for any common arity n
    (n=1 is just ``alg.mul(a, b)``)."""
    result = Element()
    for keyA, cA in a.terms.items():
        for keyB, cB in b.terms.items():
            factorsA, factorsB = _factors_of(keyA), _factors_of(keyB)
            if len(factorsA) != len(factorsB):
                raise ValueError(
                    f"tensor_mul: mismatched tensor arity ({len(factorsA)} vs {len(factorsB)})"
                )
            piece = None
            for fa, fb in zip(factorsA, factorsB):
                prod = alg.mul(Element.basis(fa), Element.basis(fb))
                piece = prod if piece is None else tensor(piece, prod)
            # Accumulate directly via add_term rather than `result = result
            # + scalar * piece`: that would rebuild (copy every existing
            # term of) `result` on every single iteration of this loop --
            # fine when result stays small, but quadratic once it grows to
            # the hundreds of terms an R-matrix has (this was the actual
            # cost behind a ~40s R-matrix intertwiner check, not the
            # R-matrix computation itself).
            scalar = cA * cB
            for k, c in piece.terms.items():
                result.add_term(k, scalar * c)
    return result


def Δ(elem: Element) -> Element:
    """The coproduct, dispatched via ``elem``'s ``.alg`` tag: ``Δ(x)`` is
    ``alg.comul(x)`` for whichever algebra ``x`` is tagged with -- so which
    algebra's coproduct rule to use is read off the argument itself, the
    same way ``*`` reads ``.alg`` off its operands to pick a product (see
    ``Element.__mul__``). Requires a tagged Element (e.g. via ``alg.elt(key)``
    or a generator accessor like ``alg.E``/``alg.K``) for the same reason
    ``*`` does: there is no algebra to call ``comul`` on otherwise --
    untagged, call ``alg.comul(elem)`` directly instead.

    The result is itself tagged with the same algebra -- even though a
    per-example ``comul()`` implementation typically builds its result
    from untagged pieces internally -- so e.g. ``Δ(a) * Δ(b)`` (the
    tensor-algebra product in H (x) H, via ``Element.__mul__``'s
    arity-aware dispatch) works directly instead of needing
    ``tensor_mul(alg, Δ(a), Δ(b))`` spelled out."""
    if elem.alg is None:
        raise TypeError(
            "Δ(elem) requires a tagged Element (e.g. via alg.elt(key) or a "
            "generator accessor like alg.E/alg.K) so it knows which algebra's "
            "comul to use -- or call alg.comul(elem) directly"
        )
    result = elem.alg.comul(elem)
    result.alg = elem.alg
    return result


def ε(elem: Element):
    """The counit, dispatched via ``elem``'s ``.alg`` tag: ``ε(x)`` is
    ``alg.counit(x)`` for whichever algebra ``x`` is tagged with -- the
    same dispatch as ``Δ``, just for the counit instead of the coproduct.
    Returns a scalar (a ``QRational``/``CycloNum``/plain number, whatever
    that algebra's ``counit`` returns), not an ``Element``. Requires a
    tagged Element for the same reason ``Δ``/``*`` do -- untagged, call
    ``alg.counit(elem)`` directly instead."""
    if elem.alg is None:
        raise TypeError(
            "ε(elem) requires a tagged Element (e.g. via alg.elt(key) or a "
            "generator accessor like alg.E/alg.K) so it knows which algebra's "
            "counit to use -- or call alg.counit(elem) directly"
        )
    return elem.alg.counit(elem)
