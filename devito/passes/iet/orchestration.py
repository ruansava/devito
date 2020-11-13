from collections import namedtuple

import cgen as c
from sympy import Or
import numpy as np

from devito.data import FULL
from devito.ir.equations import DummyEq
from devito.ir.iet import (Call, Conditional, ElementalFunction, Expression, List,
                           Iteration, PointerCast, SyncSpot, While, FindNodes,
                           LocalExpression, Transformer, derive_parameters, make_tfunc)
from devito.ir.support import Forward
from devito.passes.iet.engine import iet_pass
from devito.symbolics import CondEq, FieldFromComposite, IndexedPointer, ListInitializer
from devito.tools import as_mapper, as_list, filter_ordered, filter_sorted
from devito.types import (STDThreadArray, WaitLock, WithLock, FetchWait,
                          FetchWaitPrefetch, Delete)

__init__ = ['Orchestrator']


class Orchestrator(object):

    """
    Coordinate host and device in a heterogeneous system (e.g., CPU and GPU).
    This boils down to introducing data transfers, synchronizations, asynchronous
    computation and so on.
    """

    _Parallelizer = None
    """
    To be specified by the subclasses. This is used to generate IETs for the
    data transfers between host and device.
    """

    def __init__(self, sregistry):
        self.sregistry = sregistry

        if self._Parallelizer is None:
            raise NotImplementedError

    @property
    def _P(self):
        """Shortcut for self._Parallelizer."""
        return self._Parallelizer

    def _make_waitlock(self, iet, sync_ops, *args):
        waitloop = List(
            header=c.Comment("Wait for `%s` to be copied to the host" %
                             ",".join(s.target.name for s in sync_ops)),
            body=While(Or(*[CondEq(s.handle, 0) for s in sync_ops])),
            footer=c.Line()
        )

        iet = List(body=(waitloop,) + iet.body)

        return iet

    def _make_withlock(self, iet, sync_ops, pieces, root):
        locks = sorted({s.lock for s in sync_ops}, key=lambda i: i.name)

        # Determine the number of threads requires for `iet` with the given `sync_ops`
        lock_sizes = {i.size for i in locks}
        assert len(lock_sizes) == 1
        nthreads = lock_sizes.pop()

        # The std::thread array
        threads = STDThreadArray(name=self.sregistry.make_name(prefix='threads'),
                                 nthreads=nthreads)
        pieces.threads.append(threads)
        queueid = sum(i.size for i in pieces.threads)

        setlock = []
        actions = []
        for s in sync_ops:
            setlock.append(Expression(DummyEq(s.handle, 0)))

            imask = [s.handle.indices[d] if d.root in s.lock.locked_dimensions else FULL
                     for d in s.target.dimensions]
            actions.append(List(
                header=self._P._map_update_wait_host(s.target, imask, queueid=queueid),
                body=Expression(DummyEq(s.lock[0], 1)),
                footer=c.Line()
            ))

        # Turn `iet` into an ElementalFunction so that it can be
        # executed asynchronously by `threadhost`
        name = self.sregistry.make_name(prefix='copy_device_to_host')
        body = List(body=tuple(actions) + iet.body)
        tfunc = make_tfunc(name, body, threads, locks, root, self.sregistry)
        pieces.tfuncs.append(tfunc)

        # Replace `iet` with actions to fire up the `tfunc`
        #TODO
        #iet = List(
        #    header=c.Line(),
        #    body=setlock + [List(
        #        header=[c.Line(), c.Comment("Spawn %s to perform the copy" % thread)],
        #        body=Call('std::thread', efunc.make_call(is_indirect=True),
        #                  retobj=thread,)
        #    )]
        #)

        # Initialize the locks
        for i in locks:
            values = np.ones(i.shape, dtype=np.int32).tolist()
            pieces.init.append(LocalExpression(DummyEq(i, ListInitializer(values))))

        # Fire up the threads
        #TODO
        #body = Call('std::thread', tfunc.make_call(is_indirect=True),
        #            retobj=threads[threads.dimension])
        #threadsinit = Iteration(body, threads.dimension, threads.size - 1)
        #pieces.init.append(threadsinit)

        # Final wait before jumping back to Python land
        sdata = tfunc.sdata
        body = [Expression(DummyEq(FieldFromComposite(sdata._field_alive,
                                                      sdata[threads.dimension]), 0)),
                Call(FieldFromComposite('join', threads[threads.dimension]))]
        threadswait = Iteration(body, threads.dimension, threads.size - 1)
        pieces.finalize.append(List(
            header=c.Comment("Wait for completion of %s" % threads),
            body=threadswait
        ))
        from IPython import embed; embed()

        return iet

    def _make_fetchwait(self, iet, sync_ops, *args):
        # Construct fetches
        fetches = []
        for s in sync_ops:
            fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetches.append(self._P._map_to(s.function, imask))

        # Glue together the new IET pieces
        iet = List(header=fetches, body=iet)

        return iet

    def _make_fetchwaitprefetch(self, iet, sync_ops, pieces, *args):
        thread = STDThread(self.sregistry.make_name(prefix='thread'))
        pieces.threads.append(thread)
        queueid = len(pieces.threads)

        threadwait = Call(FieldFromComposite('join', thread))

        fetches = []
        prefetches = []
        presents = []
        for s in sync_ops:
            if s.direction is Forward:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
                fsize = s.function._C_get_field(FULL, s.dim).size
                fc_cond = fc + (s.size - 1) < fsize
                pfc = s.fetch + 1
                pfc_cond = pfc + (s.size - 1) < fsize
            else:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_max)
                fc_cond = fc >= 0
                pfc = s.fetch - 1
                pfc_cond = pfc >= 0

            # Construct fetch IET
            imask = [(fc, s.size) if d.root is s.dim.root else FULL for d in s.dimensions]
            fetch = List(header=self._P._map_to_wait(s.function, imask, queueid))
            fetches.append(Conditional(fc_cond, fetch))

            # Construct present clauses
            imask = [(s.fetch, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            presents.extend(as_list(self._P._map_present(s.function, imask)))

            # Construct prefetch IET
            imask = [(pfc, s.size) if d.root is s.dim.root else FULL
                     for d in s.dimensions]
            prefetch = List(header=self._P._map_to_wait(s.function, imask, queueid))
            prefetches.append(Conditional(pfc_cond, prefetch))

        functions = filter_ordered(s.function for s in sync_ops)
        casts = [PointerCast(f) for f in functions]

        # Turn init IET into an efunc
        name = self.sregistry.make_name(prefix='init_device')
        body = List(body=casts + fetches)
        parameters = filter_sorted(functions + derive_parameters(body))
        efunc = ElementalFunction(name, body, 'void', parameters)
        pieces.efuncs.append(efunc)

        # Call init IET
        efunc_call = efunc.make_call(is_indirect=True)
        pieces.init.append(List(
            header=c.Comment("Spawn %s to initialize data" % thread),
            body=Call('std::thread', efunc_call, retobj=thread),
            footer=c.Line()
        ))

        # Turn prefetch IET into an efunc
        name = self.sregistry.make_name(prefix='prefetch_host_to_device')
        body = List(body=casts + prefetches)
        parameters = filter_sorted(functions + derive_parameters(body))
        efunc = ElementalFunction(name, body, 'void', parameters)
        pieces.efuncs.append(efunc)

        # Call prefetch IET
        efunc_call = efunc.make_call(is_indirect=True)
        call = Call('std::thread', efunc_call, retobj=thread)

        # Glue together all the new IET pieces
        iet = List(
            header=[c.Line(),
                    c.Comment("Wait for %s to be available again" % thread)],
            body=[threadwait, List(
                header=[c.Line()] + presents,
                body=(iet, List(
                    header=[c.Line(), c.Comment("Spawn %s to prefetch data" % thread)],
                    body=call,
                    footer=c.Line()
                ),)
            )]
        )

        # Final wait before jumping back to Python land
        pieces.finalize.append(List(
            header=c.Comment("Wait for completion of %s" % thread),
            body=threadwait
        ))

        return iet

    def _make_delete(self, iet, sync_ops, *args):
        # Construct deletion clauses
        deletions = []
        for s in sync_ops:
            if s.dim.is_Custom:
                fc = s.fetch.subs(s.dim, s.dim.symbolic_min)
                imask = [(fc, s.size) if d.root is s.dim.root else FULL
                         for d in s.dimensions]
            else:
                imask = [(s.fetch, s.size) if d.root is s.dim.root else FULL
                         for d in s.dimensions]
            deletions.append(self._P._map_delete(s.function, imask))

        # Glue together the new IET pieces
        iet = List(header=c.Line(), body=iet, footer=[c.Line()] + deletions)

        return iet

    @iet_pass
    def process(self, iet):

        def key(s):
            # The SyncOps are to be processed in the following order
            return [WaitLock, WithLock, Delete, FetchWait, FetchWaitPrefetch].index(s)

        callbacks = {
            WaitLock: self._make_waitlock,
            WithLock: self._make_withlock,
            FetchWait: self._make_fetchwait,
            FetchWaitPrefetch: self._make_fetchwaitprefetch,
            Delete: self._make_delete
        }

        sync_spots = FindNodes(SyncSpot).visit(iet)

        if not sync_spots:
            return iet, {}

        pieces = namedtuple('Pieces', 'init finalize tfuncs threads')([], [], [], [])

        subs = {}
        for n in sync_spots:
            mapper = as_mapper(n.sync_ops, lambda i: type(i))
            for _type in sorted(mapper, key=key):
                subs[n] = callbacks[_type](subs.get(n, n), mapper[_type], pieces, iet)

        iet = Transformer(subs).visit(iet)

        # Add initialization and finalization code
        init = List(body=pieces.init, footer=c.Line())
        finalize = List(header=c.Line(), body=pieces.finalize)
        iet = iet._rebuild(body=(init,) + iet.body + (finalize,))

        return iet, {'efuncs': pieces.tfuncs, 'includes': ['thread']}
