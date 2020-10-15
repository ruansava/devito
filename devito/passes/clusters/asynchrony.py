from collections import OrderedDict, defaultdict
from itertools import chain

from cached_property import cached_property

from devito.ir.clusters import Queue, Cluster
from devito.ir.support import AFFINE, SEQUENTIAL, Scope
from devito.symbolics import uxreplace
from devito.tools import DefaultOrderedDict, as_tuple, filter_ordered, flatten, timed_pass
from devito.types import (Array, CustomDimension, Eq, Lock, WaitLock, SetLock, UnsetLock,
                          CopyData, DeleteData, ModuloDimension)

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
        waits = defaultdict(list)
        tasks = defaultdict(list)
        for c0 in clusters:
            if not self.key(c0):
                continue

            # Prevent future writes to interfere with a task by waiting on a lock
            may_require_lock = {i for i in c0.scope.reads if i.is_AbstractFunction}
            protected = defaultdict(set)
            for c1 in clusters:
                offset = int(clusters.index(c1) <= clusters.index(c0))

                for f in may_require_lock:
                    try:
                        writes = c1.scope.writes[f]
                    except KeyError:
                        # No read-write dependency, ignore
                        continue

                    try:
                        ld = f.indices[d]
                    except KeyError:
                        # Would degenerate to a scalar, but we rather use an Array
                        # of size 1 for simplicity
                        ld = CustomDimension(name='ld', symbolic_size=1)
                    lock = locks.setdefault(f, Lock(name='lock%d' % len(locks),
                                                    dimensions=ld))

                    for w in writes:
                        try:
                            index = w[d]
                            logical_index = index + offset
                        except TypeError:
                            assert ld.symbolic_size == 1
                            index = 0
                            logical_index = 0

                        if logical_index in protected[f]:
                            continue

                        waits[c1].append(WaitLock(lock[index]))
                        protected[f].add(logical_index)

            # Taskify `c0`
            acquired = []
            copyrelease = []
            for f in protected:
                lock = locks[f]

                try:
                    indices = sorted({r[d] for r in c0.scope.reads[f]})
                except TypeError:
                    assert lock.size == 1
                    indices = [0]

                acquired.extend([SetLock(lock[i]) for i in indices])
                copyrelease.extend([CopyData(f, {d: i}) for i in indices])
                copyrelease.extend([UnsetLock(lock[i]) for i in indices])
            tasks[c0] = acquired + copyrelease

        processed = [c.rebuild(syncs={d: waits[c] + tasks[c]}) for c in clusters]

        return processed


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
