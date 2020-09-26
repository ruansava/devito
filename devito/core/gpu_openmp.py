from collections import defaultdict
from functools import partial, singledispatch

from ctypes import c_void_p
import cgen as c
from sympy import Function
import numpy as np
import sympy

from devito.core.operator import OperatorCore
from devito.data import FULL
from devito.exceptions import InvalidOperator
from devito.ir.equations import DummyEq
from devito.ir.iet import (Block, Call, Callable, Conditional, ElementalFunction, Expression,
                           List, LocalExpression, While, FindNodes, FindSymbols,
                           MapExprStmts, Transformer, make_efunc)
from devito.logger import warning
from devito.mpi.distributed import MPICommObject
from devito.mpi.routines import (CopyBuffer, HaloUpdate, IrecvCall, IsendCall, SendRecv,
                                 MPICallable)
from devito.passes.clusters import (Lift, cire, cse, eliminate_arrays, extract_increments,
                                    factorize, fuse, optimize_pows)
from devito.passes.iet import (DataManager, Storage, Ompizer, OpenMPIteration,
                               ParallelTree, OpenMPRegion, optimize_halospots,
                               mpiize, hoist_prodders, iet_pass)
from devito.symbolics import CondEq, DefFunction, FieldFromComposite, ListInitializer
from devito.tools import as_tuple, filter_sorted, timed_pass
from devito.types import Array, CustomDimension, LocalObject

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
        self.device_fit = options['device-fit']

    @classmethod
    def _map_data(cls, f):
        if f.is_Array:
            return f.symbolic_shape
        else:
            return tuple(f._C_get_field(FULL, d).size for d in f.dimensions)

    @classmethod
    def _map_to(cls, f):
        return cls.lang['map-enter-to'](f.name, ''.join('[0:%s]' % i
                                                        for i in cls._map_data(f)))

    @classmethod
    def _map_alloc(cls, f):
        return cls.lang['map-enter-alloc'](f.name, ''.join('[0:%s]' % i
                                                           for i in cls._map_data(f)))

    @classmethod
    def _map_present(cls, f):
        raise NotImplementedError

    @classmethod
    def _map_update(cls, f):
        return cls.lang['map-update'](f.name, ''.join('[0:%s]' % i
                                                      for i in cls._map_data(f)))

    @classmethod
    def _map_update_host(cls, f, imask):
        datasize = cls._map_data(f)
        assert len(imask) == len(datasize)
        ranges = ['[%s:%s]' % ((0, j) if i is FULL else (i, ''.join(str(i+1).split())))
                  for i, j in zip(imask, datasize)]
        return cls.lang['map-update-host'](f.name, ''.join(ranges))

    @classmethod
    def _map_release(cls, f):
        return cls.lang['map-release'](f.name, ''.join('[0:%s]' % i
                                                       for i in cls._map_data(f)))

    @classmethod
    def _map_delete(cls, f):
        return cls.lang['map-exit-delete'](f.name, ''.join('[0:%s]' % i for i in
                                                           cls._map_data(f)), ' if(1%s)' %
                                           ''.join(' && (%s != 0)' % i for i in
                                                   cls._map_data(f)))

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
        for either device offloading or host parallelism.
        """
        assert candidates
        root = candidates[0]

        if is_ondevice(root, self.device_fit):
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

    def _make_orchestration(self, iet):
        partrees = FindNodes(ParallelTree).visit(iet)
        parregions = FindNodes(OpenMPRegion).visit(iet)

        wmapper = defaultdict(dict)
        rmapper = defaultdict(dict)
        for parregion in parregions:
            assert type(parregion.root) is OpenMPIteration

            indexeds = FindSymbols('indexeds').visit(parregion)
            ondevice = [i for i in indexeds if is_ondevice(i, self.device_fit)]

            # Get the Function's that, if written, will have to be copied to the host
            rimaskss = build_imask_mapper(ondevice, parregion.root)

            rmapper[parregion].update(rimaskss)

            # Get the closest tree performing a write to any of the Function's read
            # in ``parregion``.
            for i in sorted(partrees, key=lambda i: parregion.partree is i, reverse=True):
                if not isinstance(i.root, DeviceOpenMPIteration):
                    continue

                indexeds = [e.output for e in FindNodes(Expression).visit(i)]
                if not any(idx.function in set(rimaskss) for idx in indexeds):
                    continue

                # Found!
                wmapper[i].update(build_imask_mapper(indexeds, i))
                break

        locks = Locks(self.sregistry.make_name)

        # Protect `wnext` from race conditions
        mapper = {}
        for wnext, wimaskss in wmapper.items():
            conditions = []
            for f, imasks in wimaskss.items():
                for i in imasks:
                    conditions.append(CondEq(locks.makedefault(f, i), 0))
            condition = sympy.And(*conditions)
            body = List(header=c.Comment("Wait for `%s` to be copied to the host" %
                                         ",".join(f.name for f in wimaskss)),
                        body=While(condition),
                        footer=c.Line())
            mapper[wnext] = wnext._rebuild(prefix=(body,) + wnext.prefix)

        # The `parregion` will be executed by another thread asynchronously
        threadhost = STDThread('thread')
        threadwait = List(
            body=Conditional(DefFunction(FieldFromComposite('joinable', threadhost)),
                             Call(FieldFromComposite('join', threadhost)))
        )

        # Protect `parregion` from race conditions and add in pragmas for data copy
        efuncs = []
        for n, (parregion, rimaskss) in enumerate(rmapper.items()):
            copies = []
            setlock = []
            resetlock = []
            for f, imasks in rimaskss.items():
                for i in imasks:
                    lock = locks.makedefault(f, i)
                    setlock.append(Expression(DummyEq(lock, 0)))
                    copies.append(self._map_update_host(f, i))
                    resetlock.append(Expression(DummyEq(lock, 1)))

            # Turn `parregion` into an ElementalFunction so that it can be
            # executed asynchronously by `threadhost`
            header = [c.Line(), c.Comment("Device can now write again to `%s`"
                                          % ",".join(f.name for f in rimaskss))]
            body = List(header=copies + header, body=resetlock, footer=c.Line())
            efunc = make_efunc('copy_device_to_host%d' % n, List(body=[body, parregion]))
            efuncs.append(efunc)

            # Replace with a call to the `efunc` plus suitable locking logic
            ascall = List(
                header=c.Comment("Wait for host thread to be available again"),
                body=[threadwait] + [List(
                    header=c.Line(),
                    body=setlock + [List(
                        header=[c.Line(), c.Comment("Spawn thread to perform the copy")],
                        body=Call('std::thread', efunc.make_call(), retobj=threadhost)
                    )]
                )]
            )
            mapper[parregion] = ascall

        iet = Transformer(mapper).visit(iet)

        # Make sure we don't jump back to Python-land until the host thread is done
        threadwait = List(header=[c.Line(),
                                  c.Comment("Wait for completion of host thread")],
                          body=threadwait)

        # Explicitly initialize the locks
        body = []
        for i in locks.values():
            values = np.ones(i.shape, dtype=np.int32).tolist()
            body.append(LocalExpression(DummyEq(i, ListInitializer(values))))
        locks_init = List(body=body, footer=c.Line())

        iet = iet._rebuild(body=(locks_init,) + iet.body + (threadwait,))

        return iet, {'efuncs': efuncs, 'includes': ['thread']}


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
            Accepted: ['device-fit'].
            * 'device-fit': an iterable of `Function`s that are guaranteed to fit
              in the device memory. By default, all `Function`s except saved
              `TimeFunction`'s are assumed to fit in the device memory.
        """
        super().__init__(sregistry)
        self.device_fit = options['device-fit']

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
    def place_ondevice(self, iet):

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
                if is_ondevice(i, self.device_fit):
                    self._map_function_on_high_bw_mem(iet, i, storage)
            for i in filter_sorted(reads):
                if is_ondevice(i, self.device_fit):
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

    CIRE_REPEATS_SOPS = 5
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

        # CIRE
        o['min-storage'] = False
        o['cire-rotate'] = False
        o['cire-onstack'] = False
        o['cire-maxpar'] = oo.pop('cire-maxpar', True)
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
        o['gpu-direct'] = oo.pop('gpu-direct', False)

        # GPU data
        o['device-fit'] = as_tuple(oo.pop('device-fit', None))

        if oo:
            raise InvalidOperator("Unsupported optimization options: [%s]"
                                  % ", ".join(list(oo)))

        kwargs['options'].update(o)

        return kwargs

    @classmethod
    @timed_pass(name='specializing.Clusters')
    def _specialize_clusters(cls, clusters, **kwargs):
        options = kwargs['options']
        platform = kwargs['platform']
        sregistry = kwargs['sregistry']

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


class DeviceOpenMPCustomOperator(DeviceOpenMPOperator):

    _known_passes = ('optcomms', 'openmp', 'c++par', 'mpi', 'prodders', 'gpu-direct')
    _known_passes_disabled = ('blocking', 'denormals', 'simd')
    assert not (set(_known_passes) & set(_known_passes_disabled))

    @classmethod
    def _make_passes_mapper(cls, **kwargs):
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

        return super(DeviceOpenMPCustomOperator, cls)._build(expressions, **kwargs)

    @classmethod
    @timed_pass(name='specializing.IET')
    def _specialize_iet(cls, graph, **kwargs):
        options = kwargs['options']
        sregistry = kwargs['sregistry']
        passes = as_tuple(kwargs['mode'])

        # Fetch passes to be called
        passes_mapper = cls._make_passes_mapper(**kwargs)

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

def is_ondevice(maybe_symbol, device_fit):
    """
    True if all functions are allocated in the device memory, False otherwise.
    """
    # `maybe_symbol` may be an Indexed, a Function, or even an actual piece of IET
    try:
        functions = (maybe_symbol.function,)
    except AttributeError:
        assert maybe_symbol.is_Node
        functions = FindSymbols().visit(maybe_symbol)

    return all(not (f.is_TimeFunction and f.save is not None and f not in device_fit)
               for f in functions)


def build_imask_mapper(indexeds, root):
    """
    An Indexed mask, or simply imask, is a representation of an Indexed such that,
    given a Dimension `d` and the corresponding index `i`,

        * if `d` is a device-offloaded Dimension, then the mask abstracts away
          the index using the special object FULL
        * if, instead, `d` is a sequential (or simply non-offloaded) Dimension, then
          the mask will use `i`.

    This function builds a mapper from Function's to imask's for all Indexed's
    found in a given IET.

    Examples
    --------
    Let's say the caller wants to build an imask mapper for a subset of the Indexed's
    within `root`. For example, `root` is a perfect nest of Iteration's over the
    Dimension's `[t, x, y, z]`, while `indexeds = [u[t1, x, y, z], u[t2, x, y, z],
    v[t1, x, 0, z]]`. Then the imask mapper will be `{u(t, x, ,y, z): [(t1, FULL,
    FULL, FULL), (t2, FULL, FULL, FULL)], v(t, x, y, z): [(t1, FULL, FULL, FULL)]}`.
    """
    imasks = defaultdict(set)
    for indexed in indexeds:
        f = indexed.function
        if not f.is_DiscreteFunction:
            continue

        assert root.dim in f.dimensions
        n = f.dimensions.index(root.dim)
        imask = indexed.indices[:n] + (FULL,)*len(indexed.indices[n:])

        imasks[indexed.function].add(tuple(imask))

    return imasks


class STDThread(LocalObject):
    dtype = type('std::thread', (c_void_p,), {})

    def __init__(self, name):
        self.name = name

    # Pickling support
    _pickle_args = ['name']


class Locks(dict):

    """
    A simple mapper between Function's and locks to make sure we have
    a unique lock per Function.
    """

    def __init__(self, make_name, *args, **kwargs):
        self.make_name = make_name
        super().__init__(*args, **kwargs)

    def makedefault(self, f, imask):
        # The lock-protected Dimension's
        locked = [i for i in imask if i is not FULL]

        if f not in self:
            name = self.make_name(prefix='%s_lock' % f.name)

            n = len(locked)
            dims = []
            for d, s in zip(f.dimensions[:n], f.shape[:n]):
                if d.is_Stepping:
                    dims.append(CustomDimension(name=d.name, symbolic_size=s))
                else:
                    dims.append(d)

            self[f] = Array(name=name, dimensions=dims, dtype=np.int32, scope='stack',
                            volatile=True)

        return self[f][locked]
