import pytest

from devito import Grid, Function, TimeFunction, Eq, Operator


def test_basic():
    grid = Grid(shape=(4, 4))

    u = TimeFunction(name='u', grid=grid)

    op = Operator(Eq(u.forward, u + 1))
