"""Tiny shared helper for rendering exponents/indices as real Unicode
sub/superscript characters in ``pretty()`` output (e.g. ``E²`` instead of
``E^2``, ``f₁⁺`` instead of ``f1+``), plus the tensor product symbol
(``⊗`` instead of ``(x)``), used by every example's pretty-printer.

This is presentation-only and has nothing to do with any algebra's
math, which is why it lives here rather than being duplicated (or,
worse, tangled into) each example's own ``_reduce_word`` -- unlike the
normal-form reduction logic, which is deliberately *not* shared (see
CLAUDE.md's design philosophy), plain digit-to-Unicode lookup tables are
just string formatting and gain nothing from being hand-written per
example.
"""

from __future__ import annotations

_SUPER = {
    "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
    "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
    "+": "⁺", "-": "⁻", "(": "⁽", ")": "⁾", "/": "⁄",
}

_SUB = {
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
    "+": "₊", "-": "₋", "(": "₍", ")": "₎",
}

TENSOR = "⊗"  # U+2297 CIRCLED TIMES


def superscript(s) -> str:
    """Render ``s`` (typically an int or a short string like "1/2") with
    Unicode superscript characters, e.g. ``superscript(23) == "²³"``.
    Characters with no superscript form (most letters) are passed through
    unchanged."""
    return "".join(_SUPER.get(ch, ch) for ch in str(s))


def subscript(s) -> str:
    """Render ``s`` with Unicode subscript characters, e.g.
    ``subscript(1) == "₁"``. Characters with no subscript form are
    passed through unchanged."""
    return "".join(_SUB.get(ch, ch) for ch in str(s))
