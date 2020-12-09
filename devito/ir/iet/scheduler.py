from collections import OrderedDict

import numpy as np

from devito.exceptions import InvalidOperator
from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, FindNodes, FindSymbols,
                           XSubs, Transformer)
from devito.symbolics import IntDiv, uxreplace, xreplace_indices
from devito.tools import DefaultOrderedDict, as_mapper, is_integer, flatten, timed_pass
from devito.types import ConditionalDimension, ModuloDimension

__all__ = ['iet_build', 'iet_lower_dims']


@timed_pass(name='build')
def iet_build(stree):
    """
    Construct an Iteration/Expression tree(IET) from a ScheduleTree.
    """
    nsections = 0
    queues = OrderedDict()
    for i in stree.visit():
        if i == stree:
            # We hit this handle at the very end of the visit
            return List(body=queues.pop(i))

        elif i.is_Exprs:
            exprs = [Increment(e) if e.is_Increment else Expression(e) for e in i.exprs]
            body = ExpressionBundle(i.ispace, i.ops, i.traffic, body=exprs)

        elif i.is_Conditional:
            body = Conditional(i.guard, queues.pop(i))

        elif i.is_Iteration:
            body = Iteration(queues.pop(i), i.dim, i.limits, direction=i.direction,
                             properties=i.properties, uindices=i.sub_iterators)

        elif i.is_Section:
            body = Section('section%d' % nsections, body=queues.pop(i))
            nsections += 1

        elif i.is_Halo:
            body = HaloSpot(i.halo_scheme, body=queues.pop(i))

        elif i.is_Sync:
            body = SyncSpot(i.sync_ops, body=queues.pop(i))

        queues.setdefault(i.parent, []).append(body)

    assert False


@timed_pass(name='lower_dims')
def iet_lower_dims(iet):
    """
    Lower the DerivedDimensions in ``iet``.
    """
    iet = _lower_stepping_dims(iet)
    iet = _lower_conditional_dims(iet)

    return iet


def _lower_stepping_dims(iet):
    """
    Lower SteppingDimensions by turning index functions involving
    SteppingDimensions into ModuloDimensions.

    Examples
    --------
    u[t+1, x] = u[t, x] + 1

    becomes

    u[t1, x] = u[t0, x] + 1
    """
    # Replace SteppingDimensions with ModuloDimensions in each Iteration header
    subs = {}
    for iteration in FindNodes(Iteration).visit(iet):
        steppers = {d for d in iteration.uindices if d.is_Stepping}
        if not steppers:
            continue
        d = iteration.dim

        exprs = FindNodes(Expression).visit(iteration)
        symbols = flatten(e.free_symbols for e in exprs)
        indexeds = [i for i in symbols if i.is_Indexed]

        mapper = DefaultOrderedDict(lambda: DefaultOrderedDict(set))
        for indexed in indexeds:
            try:
                iaf = indexed.indices[d]
            except KeyError:
                continue

            # Sanity checks
            sis = iaf.free_symbols & steppers
            if len(sis) == 0:
                continue
            elif len(sis) == 1:
                si = sis.pop()
            else:
                raise InvalidOperator("Cannot use multiple SteppingDimensions "
                                      "to index into a Function")
            size = indexed.function.shape_allocated[d]
            assert is_integer(size)

            mapper[size][si].add(iaf)

        if not mapper:
            continue

        mds = []
        for size, v in mapper.items():
            for si, iafs in v.items():
                # Offsets are sorted so that the semantic order (t0, t1, t2) follows
                # SymPy's index ordering (t, t-1, t+1) afer modulo replacement so
                # that associativity errors are consistent. This corresponds to
                # sorting offsets {-1, 0, 1} as {0, -1, 1} assigning -inf to 0
                siafs = sorted(iafs, key=lambda i: -np.inf if i - si == 0 else (i - si))

                for iaf in siafs:
                    name = '%s%d' % (si.name, len(mds))
                    offset = uxreplace(iaf, {si: d.root})
                    mds.append(ModuloDimension(name, si, offset, size, origin=iaf))

        uindices = mds + [i for i in iteration.uindices if i not in steppers]
        subs[iteration] = iteration._rebuild(uindices=uindices)

    if subs:
        iet = Transformer(subs, nested=True).visit(iet)

    subs = {}
    for i in FindNodes(Iteration).visit(iet):
        if not i.uindices:
            # Be quick: avoid uselessy reconstructing nodes
            continue

        # In an expression, there could be `u[t+1, ...]` and `v[t+1, ...]`, where
        # `u` and `v` are TimeFunction with circular time buffers (save=None) *but*
        # different modulo extent. The `t+1` indices above are therefore conceptually
        # different, so they will be replaced with the proper ModuloDimension through
        # two different calls to `xreplace`
        mindices = [d for d in i.uindices if d.is_Modulo]
        groups = as_mapper(mindices, lambda d: d.modulo)
        root = i
        for k, v in groups.items():
            mapper = {d.origin: d for d in v}

            def rule(e):
                f = e.function
                if not (f.is_TimeFunction or f.is_Array):
                    return False
                try:
                    return f.shape_allocated[i.dim] == k
                except KeyError:
                    return False

            replacer = lambda e: xreplace_indices(e, mapper, rule)
            root = XSubs(replacer=replacer).visit(root)
        subs[i] = root

    return Transformer(subs).visit(iet)


def _lower_conditional_dims(iet):
    """
    Lower ConditionalDimensions by turning index functions involving
    ConditionalDimensions into integer-division expressions.

    Examples
    --------
    u[t_sub, x] = u[time, x]

    becomes

    u[time / 4, x] = u[time, x]
    """
    cdims = [d for d in FindSymbols('free-symbols').visit(iet)
             if isinstance(d, ConditionalDimension)]
    mapper = {d: IntDiv(d.index, d.factor) for d in cdims}
    iet = XSubs(mapper).visit(iet)

    return iet
