from collections import OrderedDict

from devito.ir.iet import (Expression, Increment, Iteration, List, Conditional, SyncSpot,
                           Section, HaloSpot, ExpressionBundle, FindNodes, FindSymbols,
                           XSubs, Transformer)
from devito.symbolics import IntDiv, xreplace_indices
from devito.tools import as_mapper, timed_pass
from devito.types import ConditionalDimension

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
    iet = _lower_conditional_dims(iet)

    return iet


def _lower_conditional_dims(iet):
    """
    Lower ConditionalDimensions: index functions involving ConditionalDimensions
    are turned into integer-division expressions.

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
