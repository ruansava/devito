from collections import OrderedDict, defaultdict
from itertools import chain

from cached_property import cached_property

from devito.ir.equations import DummyEq, LoweredEq, lower_exprs
from devito.ir.clusters import Queue, clusterize
from devito.ir.support import AFFINE, SEQUENTIAL, Scope
from devito.symbolics import uxreplace
from devito.tools import DefaultOrderedDict, as_tuple, filter_ordered, flatten, timed_pass
from devito.types import (Array, CustomDimension, Eq, Lock, WaitLock, WithLock,
                          WaitThread, SpawnThread, STDThread, ModuloDimension)

__all__ = ['Buffering']


class Buffering(Queue):

    """
    Replace Functions matching a user-provided condition with Arrays. The
    computation is then performed over such Arrays, while the buffered
    Functions are only accessed for initialization and finalization.

    The read-only Functions are not buffering candidates.

    Parameters
    ----------
    key : callable, optional
        Apply buffering iff `key(function)` gives True.

    Examples
    --------
    If we have a Cluster with the following Eq

        Eq(u[time+1, x], u[time, x] + u[time-1, x] + 1)

    Then we see that `u(time, x)` is both read and written. So it is a buffering
    candidate. Let's assume that `key(u)` is True, so we apply buffering. This
    boils down to:

        1. Introduce one Cluster with two Eqs to initialize the buffer, i.e.

            Eq(u_buf[d, x], u[d, x])
            Eq(u_buf[d-1, x], u[d-1, x])

           With the ModuloDimension `d` (a sub-iterator along `time`) starting at
           either `time.symbolic_min` (Forward direction) or `time.symbolic_max`
           (Backward direction).

        2. Introduce one Cluster with one Eq to dump the buffer back into `u`

            Eq(u[time+1, x], u_buf[d+1, x])

        3. Replace all other occurrences of `u` with `u_buf`

    So eventually we have three Clusters:

        Cluster([Eq(u_buf[d, x], u[d, x]),
                 Eq(u_buf[d-1, x], u[d-1, x])])
        Cluster([Eq(u_buf[d+1, x], u[d, x] + u[d-1, x] + 1)])
        Cluster([Eq(u[time+1, x], u_buf[d+1, x])])
    """

    def __init__(self, key=None):
        if key is None:
            self.key = lambda f: f.is_DiscreteFunction
        else:
            assert callable(key)
            self.key = lambda f: f.is_DiscreteFunction and key(f)

        super(Buffering, self).__init__()

    @timed_pass(name='buffering')
    def process(self, clusters):
        return super().process(clusters)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not all(c.properties[d] >= {SEQUENTIAL, AFFINE} for c in clusters):
            return clusters

        # Locate and classify all buffered Function accesses within the `clusters`
        accessmap = AccessMap(clusters, self.key)

        # Create the buffers
        mapper = BufferMapper([(f, Buffer(f, d, accessmap[f], n))
                               for n, f in enumerate(accessmap.functions)])

        # Create Eqs to initialize `bf` if `f` is a read-write Function
        exprs = []
        for f, b in mapper.items():
            if b.is_readonly:
                continue
            indices = list(f.dimensions)
            indices[b.index] = b.bdim
            eq = Eq(b.buffer[indices], b.function[indices])
            exprs.append(LoweredEq(lower_exprs(eq)))
        processed = list(clusterize(exprs))

        # Substitution rules to replace buffered Functions with buffers
        subs = {}
        for f, b in mapper.items():
            for a in accessmap[f].accesses:
                indices = list(a.indexed.indices)
                indices[b.index] = b.mds_mapper[indices[b.index]]
                subs[a.indexed] = b.buffer[indices]

        # Create Eqs to copy back `bf` into `f`
        lwmapper = mapper.as_lastwrite_mapper()
        for c in clusters:
            exprs = [uxreplace(e, subs) for e in c.exprs]
            processed.append(c.rebuild(exprs=exprs))

            try:
                buffereds = lwmapper[c]
            except KeyError:
                continue

            exprs = []
            for b in buffereds:
                writes = list(c.scope.writes[b.function])
                if len(writes) != 1:
                    raise NotImplementedError
                write = writes.pop()

                # Build up the copy-back expression
                indices = list(write.indexed.indices)
                indices[b.index] = b.mds_mapper[indices[b.index]]
                exprs.append(DummyEq(write.indexed, b.buffer[indices]))


            # Create a thread to perform buffer-related operations (e.g., initialization,
            # copy-back, etc.) asynchronously w.r.t. the main execution flow
            #TODO
            #TODO: replace WithThread with AcquireLock and ReleaseLock
            thread = STDThread(name='thread%d' % n)
            syncs = defaultdict(list)
            for b in buffereds:
                from IPython import embed; embed()
                syncs[b.dim].append(WithLock(b.lock[indices[b.index]]))

            # Add in the Buffer's ModuloDimensions and make sure the copy-back
            # occurs in a disjoint iteration space
            sub_iterators = filter_ordered(flatten(b.mds for b in buffereds))
            ispace = c.ispace.augment({d: sub_iterators})
            ispace = ispace.lift(ispace.next(d).dim)

            processed.append(c.rebuild(exprs=exprs, ispace=ispace, syncs=syncs))

        return processed


class BufferMapper(OrderedDict):

    def as_lastwrite_mapper(self):
        ret = DefaultOrderedDict(list)
        for b in self.values():
            ret[b.lastwrite].append(b)
        return ret


class Buffer(object):

    """
    A buffer with useful metadata attached.

    Parameters
    ----------
    function : DiscreteFunction
        The object for which a buffer is created.
    dim : Dimension
        The Dimension along which the buffer is created.
    accessv : AccessV
        All accesses involving `function`.
    n : int
        A unique identifier for this Buffer.
    """

    def __init__(self, function, dim, accessv, n):
        self.function = function
        self.dim = dim
        self.accessv = accessv

        # Determine the buffer size
        slots = {i[dim] for i in accessv.accesses}
        try:
            self.size = size = max(slots) - min(slots) + 1
        except TypeError:
            assert dim not in function.dimensions
            self.size = size = 1

        # Create the buffer
        bd = CustomDimension(name='db%d' % n, symbolic_size=size, symbolic_min=0,
                             symbolic_max=size-1)
        dimensions = list(function.dimensions)
        try:
            self.index = index = function.dimensions.index(dim)
            dimensions[index] = bd
        except ValueError:
            self.index = index = 0
            dimensions.insert(index, bd)
        self.buffer = Array(name='%sb' % function.name,
                            dimensions=dimensions,
                            dtype=function.dtype,
                            halo=function.halo)

        # Create the ModuloDimensions through which the buffer is accessed
        self.mds = [ModuloDimension(dim, i, size, name='d%d' % n)
                    for n, i in enumerate(slots)]

        # Create a lock to avoid race conditions should the buffer be accessed
        # in read and write mode by two e.g. two different threads
        self.lock = Lock(name='lock%d' % n, dimensions=bd)

    def __repr__(self):
        return "Buffer[%s,<%s:%s>]" % (self.buffer.name,
                                       self.buffer.dimensions[self.index],
                                       ','.join(str(i) for i in self.mds))

    @property
    def lastwrite(self):
        return self.accessv.lastwrite

    @property
    def is_readonly(self):
        return self.lastwrite is None

    @property
    def bdim(self):
        return self.buffer.dimensions[self.index]

    @cached_property
    def mds_mapper(self):
        return {d.offset: d for d in self.mds}


class AccessV(object):

    """
    A simple data structure capturing the accesses performed by a given Function.

    Parameters
    ----------
    function : Function
        The target Function.
    readby : list of Cluster
        All Clusters accessing `function` in read mode.
    writtenby : list of Cluster
        All Clusters accessing `function` in write mode.
    """

    def __init__(self, function, readby, writtenby):
        self.function = function
        self.readby = as_tuple(readby)
        self.writtenby = as_tuple(writtenby)

    @cached_property
    def allclusters(self):
        return filter_ordered(self.readby + self.writtenby)

    @cached_property
    def lastwrite(self):
        try:
            return self.writtenby[-1]
        except IndexError:
            return None

    @cached_property
    def accesses(self):
        return tuple(chain(*[c.scope.getreads(self.function) +
                             c.scope.getwrites(self.function) for c in self.allclusters]))


class AccessMap(OrderedDict):

    def __init__(self, clusters, key):
        writtenby = DefaultOrderedDict(list)
        readby = DefaultOrderedDict(list)
        for c in clusters:
            for f in c.scope.writes:
                if key(f):
                    writtenby[f].append(c)
            for f in c.scope.reads:
                if key(f):
                    readby[f].append(c)

        self.functions = functions = filter_ordered(list(writtenby) + list(readby))

        super().__init__([(f, AccessV(f, readby[f], writtenby[f])) for f in functions])
