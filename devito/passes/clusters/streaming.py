from collections import OrderedDict, defaultdict
from itertools import chain

from cached_property import cached_property

from devito.ir.clusters import Queue, Cluster
from devito.ir.support import AFFINE, SEQUENTIAL, Scope
from devito.symbolics import uxreplace
from devito.tools import DefaultOrderedDict, as_tuple, filter_ordered, flatten, timed_pass
from devito.types import (Array, CustomDimension, Eq, Lock, WaitLock, SetLock, UnsetLock,
                          WaitThread, WithThread, SyncData, DeleteData, STDThread,
                          ModuloDimension)

__all__ = ['Buffering', 'Asynchrony', 'Prefetching']


class Buffering(Queue):

    """
    Replace read-write Functions with Arrays. This gives the compiler more control
    over storage layout, data movement (e.g. between host and device), etc.

    Parameters
    ----------
    key : callable, optional
        A Function `f` is a buffering candidate only if `key(f)` returns True.

    Examples
    --------
    Assume we have a Cluster with the following Eq

        Eq(u[time+1, x], u[time, x] + u[time-1, x] + 1)

    Then we see that `u(time, x)` is both read and written. So it is a buffering
    candidate. Let's assume that `key(u)` is True, so we apply buffering. This
    boils down to:

        1. Introduce one Cluster with two Eqs to initialize the buffer, i.e.

            Eq(ub[d, x], u[d, x])
            Eq(ub[d-1, x], u[d-1, x])

           With the ModuloDimension `d` (a sub-iterator along `time`) starting at
           either `time.symbolic_min` (Forward direction) or `time.symbolic_max`
           (Backward direction).

        2. Introduce one Cluster with one Eq to copy the buffer back into `u`

            Eq(u[time+1, x], ub[d+1, x])

        3. Replace all other occurrences of `u` with `ub`

    So eventually we have three Clusters:

        Cluster([Eq(ub[d, x], u[d, x]),
                 Eq(ub[d-1, x], u[d-1, x])])
        Cluster([Eq(ub[d+1, x], u[d, x] + u[d-1, x] + 1)])
        Cluster([Eq(u[time+1, x], ub[d+1, x])])
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

        # Create Eqs to initialize `bf`
        exprs = []
        for f, b in mapper.items():
            if b.is_readonly:
                continue
            indices = list(f.dimensions)
            indices[b.index] = b.bdim
            exprs.append(Eq(b.buffer[indices], b.function[indices]))
        init = Cluster.from_eqns(*exprs)

        # Substitution rules to replace buffered Functions with buffers
        subs = {}
        for f, b in mapper.items():
            for a in accessmap[f].accesses:
                indices = list(a.indexed.indices)
                indices[b.index] = b.mds_mapper[indices[b.index]]
                subs[a.indexed] = b.buffer[indices]

        # Create Eqs to copy back `bf` into `f`
        processed = []
        for c in clusters:
            exprs = [uxreplace(e, subs) for e in c.exprs]
            processed.append(c.rebuild(exprs=exprs))

            exprs = []
            dump = []
            for f, b in mapper.items():
                # Compulsory copyback <=> in a guard OR last write
                test0 = c in b.accessv.writtenby and c.guards.get(d, False)
                test1 = c is b.accessv.lastwrite
                if not (test0 or test1):
                    continue

                writes = list(c.scope.writes[f])
                if len(writes) != 1:
                    raise NotImplementedError
                write = writes.pop()

                indices = b.function.dimensions
                findices = list(indices)
                findices[b.index] = write[b.index]
                bindices = list(indices)
                bindices[b.index] = b.mds_mapper[write[b.index]]

                exprs.append(Eq(b.function[findices], b.buffer[bindices]))

            # Make sure the copy-back occurs in a different iteration space than `c`'s
            dump = Cluster.from_clusters(*Cluster.from_eqns(*exprs))
            dump = dump.rebuild(ispace=dump.ispace.lift(dump.ispace.next(d).dim))

            processed.append(dump)

        return init + processed


class Asynchrony(Queue):

    """
    Create asynchronous Clusters. This boils down to tagging Clusters
    with suitable SyncOps, such as WaitLock, WithThread, etc.

    Parameters
    ----------
    key : callable, optional
        A Cluster `c` is made asynchronous only if `key(c)` returns True.
    """

    pass

#    # Create a lock to avoid race conditions should the buffer be accessed
#    # in read and write mode by two e.g. two different threads
#    self.lock = Lock(name='lock%d' % n, dimensions=bd)

#    # Construct the sync-data operation
#    sync_data.append(b.buffer[indices])
#
#    # Construct the synchronization operations
#    thread = STDThread(name='thread%d' % (len(processed) - len(clusters)))
#    sync_ops = (
#        [WaitThread(thread)] +
#        [SetLock(b.lock[indices[b.index]]) for b in buffereds] +
#        [WithThread(thread)] +
#        sync_data +
#        [UnsetLock(b.lock[indices[b.index]]) for b in buffereds] +
#            SyncData()
#         ))]
#    )
#    sync_ops 
#   syncs = defaultdict(list)
#   for b in buffereds:
#       from IPython import embed; embed()
#       syncs[b.dim].extend([
#            WaitThread(thread),
#            SetLock(b.lock[indices[b.index]]),
#            WithThread(thread, [
#                SyncData()
#            ])
#        ])


class Prefetching(Queue):

    """
    Prefetch read-only Functions. This boils down to tagging Clusters with
    prefetch and deletion SyncOps.

    Parameters
    ----------
    key : callable, optional
        A Function `f` is a prefetching candidate only if `key(f)` returns True.
    """

    pass


# Utils


class BufferMapper(OrderedDict):
    pass


class Buffer(object):

    """
    A buffer with metadata attached.

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
        slots = filter_ordered(i[dim] for i in accessv.accesses)
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
