"""User API to specify calls to custom C code."""

import sympy

from devito.types.lazy import Evaluable

__all__ = ['Call']


class Call(sympy.Expr, Evaluable):
    pass
