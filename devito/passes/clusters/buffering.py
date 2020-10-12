from collections import OrderedDict

from cached_property import cached_property

from devito.ir.equations import DummyEq, LoweredEq, lower_exprs
from devito.ir.clusters import Queue, clusterize
from devito.ir.support import SEQUENTIAL, Scope
from devito.symbolics import uxreplace
from devito.tools import DefaultOrderedDict, filter_ordered, flatten, timed_pass
from devito.types import Array, CustomDimension, Eq, Lock, ModuloDimension

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
        super().process(clusters)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not all(SEQUENTIAL in c.properties[d] for c in clusters):
            return clusters

        # Locate and classify all buffered Function accesses within the `clusters`
        lastwrite = DefaultOrderedDict(lambda: None)
        readby = DefaultOrderedDict(list)
        for c in clusters:
            for f in c.scope.writes:
                if self.key(f):
                    lastwrite[f] = c
            for f in c.scope.reads:
                if self.key(f):
                    readby[f].append(c)

        # Create the buffers
        functions = filter_ordered(list(lastwrite) + list(readby))
        mapper = BufferMapper([(f, Buffer(f, d, lastwrite[f], readby[f], n))
                               for n, f in enumerate(functions)])

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

        # Make up the substitution rules to replace buffered Functions with buffers
        subs = {}
        for c in clusters:
            for f, b in mapper.items():
                for a in c.scope.getreads(f) + c.scope.getwrites(f):
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

                indices = list(write.indexed.indices)
                indices[b.index] = b.mds_mapper[indices[b.index]]
                exprs.append(DummyEq(write.indexed, b.buffer[indices]))

            sub_iterators = filter_ordered(flatten(b.mds for b in buffereds))
            ispace = c.ispace.augment({d: sub_iterators})

            processed.append(c.rebuild(exprs=exprs, ispace=ispace))

        from IPython import embed; embed()


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
    lastwrite : Cluster
        The last Cluster (in program order) performing a write to `function`.
    readby : list of Cluster
        All Clusters reading `function`.
    n : int
        A unique identifier for this Buffer.
    """

    def __init__(self, function, dim, lastwrite, readby, n):
        self.function = function
        self.dim = dim
        self.lastwrite = lastwrite
        self.readby = readby

        # Determine the buffer size
        slots = set()
        for c in [lastwrite] + readby:
            accesses = c.scope.getreads(function) + c.scope.getwrites(function)
            slots.update([i[dim] for i in accesses])
        self.size = size = len(slots)

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

        # Create a lock to avoid race conditions when accessing the buffer
        self.lock = Lock(name='lock%d' % n, dimensions=bd)

    def __repr__(self):
        return "Buffer[%s,<%s:%s>]" % (self.buffer.name,
                                       self.buffer.dimensions[self.index],
                                       ','.join(str(i) for i in self.mds))

    @property
    def is_readonly(self):
        return self.lastwrite is None

    @property
    def is_readwrite(self):
        #TODO
        pass

    @property
    def bdim(self):
        return self.buffer.dimensions[self.index]

    @cached_property
    def mds_mapper(self):
        return {d.offset: d for d in self.mds}

    def at(self, v, flag=False):
        """
        Indexify the buffer along the buffer Dimension using `v` as index.
        If `flag=True`, then indexify the buffered Function.
        """
        f = self.function if flag else self.buffer
        return f[indices]
