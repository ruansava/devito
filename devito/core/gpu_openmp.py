from collections import namedtuple
from functools import partial, singledispatch

import cgen as c
from sympy import And, Function
import numpy as np

from devito.core.cpu import CustomOperator
from devito.core.operator import OperatorCore
from devito.data import FULL
from devito.exceptions import InvalidOperator
from devito.ir.equations import DummyEq
from devito.ir.iet import (Block, Call, Callable, Conditional, ElementalFunction,
                           Expression, List, PointerCast, SyncSpot, While, FindNodes,
                           FindSymbols, LocalExpression, MapExprStmts, Transformer,
                           derive_parameters, make_efunc)
from devito.ir.support import Forward
from devito.logger import warning
from devito.mpi.distributed import MPICommObject
from devito.mpi.routines import (CopyBuffer, HaloUpdate, IrecvCall, IsendCall, SendRecv,
                                 MPICallable)
from devito.passes.equations import collect_derivatives, buffering
from devito.passes.clusters import (Lift, Stream, Tasker, cire, cse, eliminate_arrays,
                                    extract_increments, factorize, fuse, optimize_pows)
from devito.passes.iet import (DataManager, Storage, Ompizer, OpenMPIteration,
                               ParallelTree, optimize_halospots, mpiize, hoist_prodders,
                               iet_pass)
from devito.symbolics import (Byref, CondEq, DefFunction, FieldFromComposite,
                              ListInitializer, ccode)
from devito.tools import as_mapper, as_tuple, filter_sorted, timed_pass
from devito.types import Symbol, STDThread, WaitLock, WithLock, WaitAndFetch

__all__ = ['DeviceOpenMPNoopOperator', 'DeviceOpenMPOperator',
           'DeviceOpenMPCustomOperator']


class DeviceOpenMPIteration(OpenMPIteration):

    @classmethod
    def _make_construct(cls, **kwargs):
        return 'omp target teams distribute parallel for device(devicenum)'

    @classmethod
    def _make_clauses(cls, **kwargs):
        kwargs['chunk_size'] = False
        return super(DeviceOpenMPIteration, cls)._make_clauses(**kwargs)


class DeviceOmpizer(Ompizer):

    lang = dict(Ompizer.lang)
    lang.update({
        'map-enter-to': lambda i, j:
            c.Pragma('omp target enter data map(to: %s%s) device(devicenum)' % (i, j)),
        'map-enter-alloc': lambda i, j:
            c.Pragma('omp target enter data map(alloc: %s%s) device(devicenum)' % (i, j)),
        'map-update': lambda i, j:
            c.Pragma('omp target update from(%s%s) device(devicenum)' % (i, j)),
        'map-update-host': lambda i, j:
            c.Pragma('omp target update from(%s%s) device(devicenum)' % (i, j)),
        'map-update-device': lambda i, j:
            c.Pragma('omp target update to(%s%s) device(devicenum)' % (i, j)),
        'map-release': lambda i, j:
            c.Pragma('omp target exit data map(release: %s%s) device(devicenum)'
                     % (i, j)),
        'map-exit-delete': lambda i, j, k:
            c.Pragma('omp target exit data map(delete: %s%s) device(devicenum)%s'
                     % (i, j, k)),
    })

    _Iteration = DeviceOpenMPIteration

    def __init__(self, sregistry, options, key=None):
        super().__init__(sregistry, options, key=key)
        self.gpu_fit = options['gpu-fit']

    @classmethod
    def _make_sections_from_imask(cls, f, imask):
        datasize = cls._map_data(f)
        if imask is None:
            imask = [FULL]*len(datasize)
        assert len(imask) == len(datasize)
        sections = ['[%s:%s]' % ((0, j) if i is FULL else (ccode(i), 1))
                    for i, j in zip(imask, datasize)]
        return ''.join(sections)

    @classmethod
    def _map_data(cls, f):
        if f.is_Array:
            return f.symbolic_shape
        else:
            return tuple(f._C_get_field(FULL, d).size for d in f.dimensions)

    @classmethod
    def _map_to(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-enter-to'](f.name, sections)

    @classmethod
    def _map_alloc(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-enter-alloc'](f.name, sections)

    @classmethod
    def _map_present(cls, f):
        raise NotImplementedError

    @classmethod
    def _map_update(cls, f):
        return cls.lang['map-update'](f.name, ''.join('[0:%s]' % i
                                                      for i in cls._map_data(f)))

    @classmethod
    def _map_update_host(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-update-host'](f.name, sections)

    @classmethod
    def _map_update_device(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        return cls.lang['map-update-device'](f.name, sections)

    @classmethod
    def _map_release(cls, f):
        return cls.lang['map-release'](f.name, ''.join('[0:%s]' % i
                                                       for i in cls._map_data(f)))

    @classmethod
    def _map_delete(cls, f, imask=None):
        sections = cls._make_sections_from_imask(f, imask)
        # This ugly condition is to avoid a copy-back when, due to
        # domain decomposition, the local size of a Function is 0, which
        # would cause a crash
        cond = ' if(%s)' % ' && '.join('(%s != 0)' % i for i in cls._map_data(f))
        return cls.lang['map-exit-delete'](f.name, sections, cond)

    @classmethod
    def _map_pointers(cls, f):
        raise NotImplementedError

    def _make_threaded_prodders(self, partree):
        if isinstance(partree.root, DeviceOpenMPIteration):
            # no-op for now
            return partree
        else:
            return super()._make_threaded_prodders(partree)

    def _make_partree(self, candidates, nthreads=None):
        """
        Parallelize the `candidates` Iterations attaching suitable OpenMP pragmas
        for parallelism. In particular:

            * All parallel Iterations not *writing* to a host Function, that
              is a Function `f` such that ``is_on_gpu(f) == False`, are offloaded
              to the device.
            * The remaining ones, that is those writing to a host Function,
              are parallelized on the host.
        """
        assert candidates
        root = candidates[0]

        if is_on_gpu(root, self.gpu_fit, only_writes=True):
            # The typical case: all accessed Function's are device Function's, that is
            # all Function's are in the device memory. Then we offload the candidate
            # Iterations to the device

            # Get the collapsable Iterations
            collapsable = self._find_collapsable(root, candidates)
            ncollapse = 1 + len(collapsable)

            body = self._Iteration(ncollapse=ncollapse, **root.args)
            partree = ParallelTree([], body, nthreads=nthreads)
            collapsed = [partree] + collapsable

            return root, partree, collapsed
        else:
            # Resort to host parallelism
            return super()._make_partree(candidates, nthreads)

    def _make_parregion(self, partree, *args):
        if isinstance(partree.root, DeviceOpenMPIteration):
            # no-op for now
            return partree
        else:
            return super()._make_parregion(partree, *args)

    def _make_guard(self, parregion, *args):
        partrees = FindNodes(ParallelTree).visit(parregion)
        if any(isinstance(i.root, DeviceOpenMPIteration) for i in partrees):
            # no-op for now
            return parregion
        else:
            return super()._make_guard(parregion, *args)

    def _make_nested_partree(self, partree):
        if isinstance(partree.root, DeviceOpenMPIteration):
            # no-op for now
            return partree
        else:
            return super()._make_nested_partree(partree)

    @iet_pass
    def make_orchestration(self, iet):
        """
        Coordinate host and device parallelism. This pass:

            * Implements smart transfer of data between host and device
              exploiting the information carried by the IET produced by the
              previous compilation passes;
            * Creates asynchrnous tasks, implemented via C++ threads, which
              are responsible for carrying out the data transfer
            * Generates lock-based statements to implement synchronization
              between the main thread and the asynchronous tasks.
        """

        def key(s):
            # The SyncOps are to be processed in the following order:
            # 1) WithLock, 2) WaitLock, 3) WaitAndFetch
            return {WaitLock: 0, WithLock: 1, WaitAndFetch: 2}[s]

        callbacks = {
            WaitLock: self._make_orchestration_waitlock,
            WithLock: self._make_orchestration_withlock,
            WaitAndFetch: self._make_orchestration_waitandfetch
        }

        sync_spots = FindNodes(SyncSpot).visit(iet)

        if not sync_spots:
            return iet, {}

        pieces = namedtuple('Pieces', 'init finalize efuncs')([], [], [])

        subs = {}
        for n in sync_spots:
            mapper = as_mapper(n.sync_ops, lambda i: type(i))
            for _type in sorted(mapper, key=key):
                subs[n] = callbacks[_type](subs.get(n, n), mapper[_type], pieces)

        iet = Transformer(subs).visit(iet)

        # Add initialization and finalization code
        init = List(body=pieces.init, footer=c.Line())
        finalize = List(header=c.Line(), body=pieces.finalize)
        iet = iet._rebuild(body=(init,) + iet.body + (finalize,))

        return iet, {'efuncs': pieces.efuncs, 'includes': ['thread']}

    def _make_orchestration_waitlock(self, iet, sync_ops, pieces):
        waitloop = List(
            header=c.Comment("Wait for `%s` to be copied to the host" %
                             ",".join(s.target.name for s in sync_ops)),
            body=While(And(*[CondEq(s.handle, 0) for s in sync_ops])),
            footer=c.Line()
        )

        iet = List(body=(waitloop,) + iet.body)

        return iet

    def _make_orchestration_withlock(self, iet, sync_ops, pieces):
        thread = STDThread(self.sregistry.make_name(prefix='thread'))
        threadwait = List(
            body=Conditional(DefFunction(FieldFromComposite('joinable', thread)),
                             Call(FieldFromComposite('join', thread)))
        )

        setlock = []
        actions = []
        for s in sync_ops:
            setlock.append(Expression(DummyEq(s.handle, 0)))

            imask = [s.handle.indices[d] if d.root in s.lock.locked_dimensions else FULL
                     for d in s.target.dimensions]
            actions.append(List(
                header=self._map_update_host(s.target, imask),
                body=Expression(DummyEq(s.handle, 1))
            ))

        # Turn `iet` into an ElementalFunction so that it can be
        # executed asynchronously by `threadhost`
        name = self.sregistry.make_name(prefix='copy_device_to_host')
        body = (List(body=actions, footer=c.Line()),) + iet.body
        efunc = make_efunc(name, List(body=body))
        pieces.efuncs.append(efunc)

        # Replace `iet` with a call to `efunc` plus locking
        iet = List(
            header=c.Comment("Wait for %s to be available again" % thread),
            body=[threadwait] + [List(
                header=c.Line(),
                body=setlock + [List(
                    header=[c.Line(), c.Comment("Spawn %s to perform the copy" % thread)],
                    body=Call('std::thread', efunc.make_call(is_indirect=True),
                              retobj=thread,)
                )]
            )]
        )

        # Initialize the locks
        locks = sorted({s.lock for s in sync_ops}, key=lambda i: i.name)
        for i in locks:
            values = np.ones(i.shape, dtype=np.int32).tolist()
            pieces.init.append(LocalExpression(DummyEq(i, ListInitializer(values))))

        # Final wait before jumping back to Python land
        pieces.finalize.append(List(
            header=c.Comment("Wait for completion of %s" % thread),
            body=threadwait
        ))

        return iet

    def _make_orchestration_waitandfetch(self, iet, sync_ops, pieces):
        thread = STDThread(self.sregistry.make_name(prefix='thread'))
        threadwait = Call(FieldFromComposite('join', thread))

        fetches = []
        prefetches = []
        presents = []
        deletions = []
        for s in sync_ops:
            f = s.function
            for i in s.fetch:
                if s.direction is Forward:
                    fc = i.subs(s.dim, s.dim.symbolic_min)
                    fc_cond = fc <= s.dim.symbolic_max
                    pfc = i + 1
                    pfc_cond = pfc <= s.dim.symbolic_max
                else:
                    fc = i.subs(s.dim, s.dim.symbolic_max)
                    fc_cond = fc >= s.dim.symbolic_min
                    pfc = i - 1
                    pfc_cond = pfc >= s.dim.symbolic_min

                # Construct fetch IET
                imask = [fc if s.dim in d._defines else FULL for d in s.dimensions]
                fetch = List(header=self._map_to(f, imask))
                fetches.append(Conditional(fc_cond, fetch))

                # Construct deletion and present clauses
                imask = [i if s.dim in d._defines else FULL for d in s.dimensions]
                deletions.append(self._map_delete(f, imask))
                presents.append(self._map_present(f, imask))

                # Construct prefetch IET
                imask = [pfc if s.dim in d._defines else FULL for d in s.dimensions]
                prefetch = List(header=self._map_to(f, imask))
                prefetches.append(Conditional(pfc_cond, prefetch))

        functions = [s.function for s in sync_ops]
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

        # Put together all the new IET pieces
        iet = List(
            header=[c.Line(),
                    c.Comment("Wait for %s to be available again" % thread)],
            body=[threadwait, List(
                header=[c.Line()] + presents,
                body=iet.body + (List(
                    header=([c.Line()] + deletions +
                            [c.Line(), c.Comment("Spawn %s to prefetch data" % thread)]),
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


class DeviceOpenMPDataManager(DataManager):

    _Parallelizer = DeviceOmpizer

    def __init__(self, sregistry, options):
        """
        Parameters
        ----------
        sregistry : SymbolRegistry
            The symbol registry, to quickly access the special symbols that may
            appear in the IET (e.g., `sregistry.threadid`, that is the thread
            Dimension, used by the DataManager for parallel memory allocation).
        options : dict
            The optimization options.
            Accepted: ['gpu-fit'].
            * 'gpu-fit': an iterable of `Function`s that are guaranteed to fit
              in the device memory. By default, all `Function`s except saved
              `TimeFunction`'s are assumed to fit in the device memory.
        """
        super().__init__(sregistry)
        self.gpu_fit = options['gpu-fit']

    def _alloc_array_on_high_bw_mem(self, site, obj, storage):
        _storage = Storage()
        super()._alloc_array_on_high_bw_mem(site, obj, _storage)

        allocs = _storage[site].allocs + [self._Parallelizer._map_alloc(obj)]
        frees = [self._Parallelizer._map_delete(obj)] + _storage[site].frees
        storage.update(obj, site, allocs=allocs, frees=frees)

    def _map_function_on_high_bw_mem(self, site, obj, storage, read_only=False):
        """
        Place a Function in the high bandwidth memory.
        """
        alloc = self._Parallelizer._map_to(obj)

        if read_only is False:
            free = c.Collection([self._Parallelizer._map_update(obj),
                                 self._Parallelizer._map_release(obj)])
        else:
            free = self._Parallelizer._map_delete(obj)

        storage.update(obj, site, allocs=alloc, frees=free)

    @iet_pass
    def place_ondevice(self, iet, **kwargs):

        @singledispatch
        def _place_ondevice(iet):
            return iet

        @_place_ondevice.register(Callable)
        def _(iet):
            # Collect written and read-only symbols
            writes = set()
            reads = set()
            for i, v in MapExprStmts().visit(iet).items():
                if not i.is_Expression:
                    # No-op
                    continue
                if not any(isinstance(j, self._Parallelizer._Iteration) for j in v):
                    # Not an offloaded Iteration tree
                    continue
                if i.write.is_DiscreteFunction:
                    writes.add(i.write)
                reads = (reads | {r for r in i.reads if r.is_DiscreteFunction}) - writes

            # Populate `storage`
            storage = Storage()
            for i in filter_sorted(writes):
                if is_on_gpu(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage)
            for i in filter_sorted(reads):
                if is_on_gpu(i, self.gpu_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage, read_only=True)

            iet = self._dump_storage(iet, storage)

            return iet

        @_place_ondevice.register(ElementalFunction)
        def _(iet):
            return iet

        @_place_ondevice.register(CopyBuffer)
        @_place_ondevice.register(SendRecv)
        @_place_ondevice.register(HaloUpdate)
        def _(iet):
            return iet

        iet = _place_ondevice(iet)

        return iet, {}


@iet_pass
def initialize(iet, **kwargs):
    """
    Initialize the OpenMP environment.
    """
    devicenum = Symbol(name='devicenum')

    @singledispatch
    def _initialize(iet):
        comm = None

        for i in iet.parameters:
            if isinstance(i, MPICommObject):
                comm = i
                break

        if comm is not None:
            rank = Symbol(name='rank')
            rank_decl = LocalExpression(DummyEq(rank, 0))
            rank_init = Call('MPI_Comm_rank', [comm, Byref(rank)])

            ngpus = Symbol(name='ngpus')
            call = Function('omp_get_num_devices')()
            ngpus_init = LocalExpression(DummyEq(ngpus, call))

            devicenum_init = LocalExpression(DummyEq(devicenum, rank % ngpus))

            body = [rank_decl, rank_init, ngpus_init, devicenum_init]

            init = List(header=c.Comment('Begin of OpenMP+MPI setup'),
                        body=body,
                        footer=(c.Comment('End of OpenMP+MPI setup'), c.Line()))
        else:
            devicenum_init = LocalExpression(DummyEq(devicenum, 0))
            body = [devicenum_init]

            init = List(header=c.Comment('Begin of OpenMP setup'),
                        body=body,
                        footer=(c.Comment('End of OpenMP setup'), c.Line()))

        iet = iet._rebuild(body=(init,) + iet.body)

        return iet

    @_initialize.register(ElementalFunction)
    @_initialize.register(MPICallable)
    def _(iet):
        return iet

    iet = _initialize(iet)

    return iet, {'args': devicenum}


@iet_pass
def mpi_gpu_direct(iet, **kwargs):
    """
    Modify MPI Callables to enable multiple GPUs performing GPU-Direct communication.
    """
    mapper = {}
    for node in FindNodes((IsendCall, IrecvCall)).visit(iet):
        header = c.Pragma('omp target data use_device_ptr(%s) device(devicenum)' %
                          node.arguments[0].name)
        mapper[node] = Block(header=header, body=node)

    iet = Transformer(mapper).visit(iet)

    return iet, {}


class DeviceOpenMPNoopOperator(OperatorCore):

    CIRE_REPEATS_INV = 2
    """
    Number of CIRE passes to detect and optimize away Dimension-invariant expressions.
    """

    CIRE_REPEATS_SOPS = 7
    """
    Number of CIRE passes to detect and optimize away redundant sum-of-products.
    """

    CIRE_MINCOST_INV = 50
    """
    Minimum operation count of a Dimension-invariant aliasing expression to be
    optimized away. Dimension-invariant aliases are lifted outside of one or more
    invariant loop(s), so they require tensor temporaries that can be potentially
    very large (e.g., the whole domain in the case of time-invariant aliases).
    """

    CIRE_MINCOST_SOPS = 10
    """
    Minimum operation count of a sum-of-product aliasing expression to be optimized away.
    """

    PAR_CHUNK_NONAFFINE = 3
    """
    Coefficient to adjust the chunk size in non-affine parallel loops.
    """

    @classmethod
    def _normalize_kwargs(cls, **kwargs):
        o = {}
        oo = kwargs['options']

        # Execution modes
        o['mpi'] = oo.pop('mpi')

        # Strictly unneccesary, but make it clear that this Operator *will*
        # generate OpenMP code, bypassing any `openmp=False` provided in
        # input to Operator
        oo.pop('openmp')

        # Buffering
        o['buf-async-degree'] = oo.pop('buf-async-degree', None)

        # CIRE
        o['min-storage'] = False
        o['cire-rotate'] = False
        o['cire-onstack'] = False
        o['cire-maxpar'] = oo.pop('cire-maxpar', True)
        o['cire-maxalias'] = oo.pop('cire-maxalias', False)
        o['cire-repeats'] = {
            'invariants': oo.pop('cire-repeats-inv', cls.CIRE_REPEATS_INV),
            'sops': oo.pop('cire-repeats-sops', cls.CIRE_REPEATS_SOPS)
        }
        o['cire-mincost'] = {
            'invariants': oo.pop('cire-mincost-inv', cls.CIRE_MINCOST_INV),
            'sops': oo.pop('cire-mincost-sops', cls.CIRE_MINCOST_SOPS)
        }

        # GPU parallelism
        o['par-collapse-ncores'] = 1  # Always use a collapse clause
        o['par-collapse-work'] = 1  # Always use a collapse clause
        o['par-chunk-nonaffine'] = oo.pop('par-chunk-nonaffine', cls.PAR_CHUNK_NONAFFINE)
        o['par-dynamic-work'] = np.inf  # Always use static scheduling
        o['par-nested'] = np.inf  # Never use nested parallelism
        o['gpu-direct'] = oo.pop('gpu-direct', True)

        # GPU data
        o['gpu-fit'] = as_tuple(oo.pop('gpu-fit', None))

        if oo:
            raise InvalidOperator("Unsupported optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs

    @classmethod
    @timed_pass(name='specializing.Expressions')
    def _specialize_exprs(cls, expressions, **kwargs):
        options = kwargs['options']

        expressions = collect_derivatives(expressions)

        # Replace host Functions with Arrays, used as device buffers for
        # streaming-in and -out of data
        def callback(f):
            if not is_on_gpu(f, options['gpu-fit']):
                return [f.time_dim]
            else:
                return None
        expressions = buffering(expressions, callback, options)

        return expressions

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        # This callback is used to trigger certain passes only on Clusters
        # accessing at least one host Function
        key = lambda f: not is_on_gpu(f, options['gpu-fit'])

        # Identify asynchronous tasks
        clusters = Tasker(key).process(clusters)

        # Toposort+Fusion (the former to expose more fusion opportunities)
        clusters = fuse(clusters, toposort=True)

        # Hoist and optimize Dimension-invariant sub-expressions
        clusters = cire(clusters, 'invariants', sregistry, options, platform)
        clusters = Lift().process(clusters)

        # Reduce flops (potential arithmetic alterations)
        clusters = extract_increments(clusters, sregistry)
        clusters = cire(clusters, 'sops', sregistry, options, platform)
        clusters = factorize(clusters)
        clusters = optimize_pows(clusters)

        # Reduce flops (no arithmetic alterations)
        clusters = cse(clusters, sregistry)

        # Lifting may create fusion opportunities, which in turn may enable
        # further optimizations
        clusters = fuse(clusters)
        clusters = eliminate_arrays(clusters)

        # Place prefetch SyncOps
        clusters = Stream(key).process(clusters)

        return clusters

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # GPU parallelism via OpenMP offloading
        DeviceOmpizer(sregistry, options).make_parallel(graph)

        # Symbol definitions
        data_manager = DeviceOpenMPDataManager(sregistry, options)
        data_manager.place_ondevice(graph)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        # Initialize OpenMP environment
        initialize(graph)

        return graph


class DeviceOpenMPOperator(DeviceOpenMPNoopOperator):

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        # Distributed-memory parallelism
        optimize_halospots(graph)
        if options['mpi']:
            mpiize(graph, mode=options['mpi'])

        # GPU parallelism via OpenMP offloading
        DeviceOmpizer(sregistry, options).make_parallel(graph)

        # Misc optimizations
        hoist_prodders(graph)

        # Symbol definitions
        data_manager = DeviceOpenMPDataManager(sregistry, options)
        data_manager.place_ondevice(graph)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        # Initialize OpenMP environment
        initialize(graph)
        # TODO: This should be moved right below the `mpiize` pass, but currently calling
        # `mpi_gpu_direct` before Symbol definitions` block would create Blocks before
        # creating C variables. That would lead to MPI_Request variables being local to
        # their blocks. This way, it would generate incorrect C code.
        if options['gpu-direct']:
            mpi_gpu_direct(graph)

        return graph


class DeviceOpenMPCustomOperator(DeviceOpenMPOperator, CustomOperator):

    @classmethod
    def _make_exprs_passes_mapper(cls, **kwargs):
        return {
            'collect-derivs': collect_derivatives,
            'buffering': buffering
        }

    @classmethod
    def _make_clusters_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

        return {
            'factorize': factorize,
            'fuse': fuse,
            'lift': lambda i: Lift().process(cire(i, 'invariants', sregistry,
                                                  options, platform)),
            'cire-sops': lambda i: cire(i, 'sops', sregistry, options, platform),
            'cse': lambda i: cse(i, sregistry),
            'opt-pows': optimize_pows,
            'topofuse': lambda i: fuse(i, toposort=True)
        }

    @classmethod
    def _make_iet_passes_mapper(cls, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']

        ompizer = DeviceOmpizer(sregistry, options)

        return {
            'optcomms': partial(optimize_halospots),
            'openmp': partial(ompizer.make_parallel),
            'mpi': partial(mpiize, mode=options['mpi']),
            'prodders': partial(hoist_prodders),
            'gpu-direct': partial(mpi_gpu_direct)
        }

    _known_passes = (
        # Expressions
        'collect-deriv', 'buffering',
        # Clusters
        'factorize', 'fuse', 'lift', 'cire-sops', 'cse', 'opt-pows', 'topofuse',
        # IET
        'optcomms', 'openmp', 'mpi', 'prodders', 'gpu-direct'
    )
    _known_passes_disabled = ('blocking', 'denormals', 'simd')
    assert not (set(_known_passes) & set(_known_passes_disabled))

    @classmethod
    def _build(cls, expressions, **kwargs):
        # Sanity check
        passes = as_tuple(kwargs['mode'])
        for i in passes:
            if i not in cls._known_passes:
                if i in cls._known_passes_disabled:
                    warning("Got explicit pass `%s`, but it's unsupported on an "
                            "Operator of type `%s`" % (i, str(cls)))
                else:
                    raise InvalidOperator("Unknown pass `%s`" % i)

        return super()._build(expressions, **kwargs)

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_iet_passes_mapper(**kwargs)

        # Call passes
        for i in passes:
            try:
                passes_mapper[i](graph)
            except KeyError:
                pass

        # Force-call `mpi` if requested via global option
        if 'mpi' not in passes and options['mpi']:
            passes_mapper['mpi'](graph)

        # GPU parallelism via OpenMP offloading
        if 'openmp' not in passes:
            passes_mapper['openmp'](graph)

        # Symbol definitions
        data_manager = DeviceOpenMPDataManager(sregistry, options)
        data_manager.place_ondevice(graph)
        data_manager.place_definitions(graph)
        data_manager.place_casts(graph)

        # Initialize OpenMP environment
        initialize(graph)

        return graph


# Utils

def is_on_gpu(maybe_symbol, gpu_fit, only_writes=False):
    """
    True if all Function's are allocated in the device memory, False otherwise.
    Parameters
    ----------
    maybe_symbol : Indexed or Function or Node
        The inspected object. May be a single Indexed or Function, or even an
        entire piece of IET.
    gpu_fit : list of Function
        The Function's which are known to definitely fit in the device memory. This
        information is given directly by the user through the compiler option
        `gpu-fit` and is propagated down here through the various stages of lowering.
    only_writes : boo, optional
        Only makes sense if `maybe_symbol` is an IET. If True, ignore all Function's
        that do not appear on the LHS of at least one Expression. Defaults to False.
    """
    try:
        functions = (maybe_symbol.function,)
    except AttributeError:
        assert maybe_symbol.is_Node
        iet = maybe_symbol
        functions = set(FindSymbols().visit(iet))
        if only_writes:
            expressions = FindNodes(Expression).visit(iet)
            functions &= {i.write for i in expressions}

    return all(not (f.is_TimeFunction and f.save is not None and f not in gpu_fit)
               for f in functions)
