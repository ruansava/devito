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

__all__ = ['Asynchrony', 'Prefetching']


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


