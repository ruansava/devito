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

__all__ = ['Tasker', 'Prefetching']


class Tasker(Queue):

    """
    Create asynchronous Clusters, or "tasks".

    Parameters
    ----------
    key : callable, optional
        A Cluster `c` becomes asynchronous iff `key(c)` returns True.

    Notes
    -----
    From an implementation viewpoint, an asynchronous Cluster is a Cluster
    with attached suitable SyncOps, such as WaitLock, WithThread, etc.
    """

    def __init__(self, key):
        assert callable(key)
        self.key = key
        super().__init__()

    @timed_pass(name='tasker')
    def process(self, clusters):
        return super().process(clusters)

    def callback(self, clusters, prefix):
        if not prefix:
            return clusters

        d = prefix[-1].dim

        if not all(c.properties[d] >= {SEQUENTIAL, AFFINE} for c in clusters):
            return clusters

        locks = {}
        mapper = defaultdict(list)
        for c0 in clusters:
            if not self.key(c0):
                continue

            # Prevent future writes to any of the Functions read by `c0`
            # by waiting on a lock
            may_require_lock = {i for i in c0.scope.reads if i.is_AbstractFunction}
            protected_indices = set()
            for c1 in clusters:
                if c0 is c1:
                    continue

                require_lock = may_require_lock & set(c1.scope.writes)
                for i in require_lock:
                    lock = locks.setdefault(i, Lock(name='lock%d' % len(locks),
                                                    dimensions=i.indices[d]))
                    for w in c1.scope.writes[i]:
                        index = w[d]
                        if index in protected_indices:
                            # A `wait` already added in one of the previous Clusters
                            continue

                        mapper[c1].append(WaitLock(lock[index]))
                        protected_indices.add(index)

        return clusters

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


class AccessMapper(OrderedDict):

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
