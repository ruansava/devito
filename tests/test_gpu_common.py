import pytest
import numpy as np

from devito import (Constant, Eq, Grid, Function, ConditionalDimension, SubDomain,
                    TimeFunction, Operator)
from devito.archinfo import get_gpu_info
from devito.ir import Expression, Section, FindNodes, FindSymbols, retrieve_iteration_tree
from devito.passes import OpenMPIteration
from devito.types import Lock

from conftest import skipif

pytestmark = skipif(['nodevice'], whole_module=True)


class TestGPUInfo(object):

    def test_get_gpu_info(self):
        info = get_gpu_info()
        assert 'tesla' in info['architecture'].lower()


class TestCodeGeneration(object):

    def test_maxpar_option(self):
        """
        Make sure the `cire-maxpar` option is set to True by default.
        """
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, space_order=2)

        eq = Eq(u.forward, u.dy.dy)

        op = Operator(eq)

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        assert trees[0][0] is trees[1][0]
        assert trees[0][1] is not trees[1][1]


class Bundle(SubDomain):
    """
    We use this SubDomain to enforce Eqs to end up in different loops.
    """

    name = 'bundle'

    def define(self, dimensions):
        x, y, z = dimensions
        return {x: ('middle', 0, 0), y: ('middle', 0, 0), z: ('middle', 0, 0)}


class TestStreaming(object):

    def test_buffering_in_isolation(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)

        eqn = Eq(u.forward, u + 1)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt='buffering')

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 2
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1
        b = buffers.pop()
        assert b.symbolic_shape[0] == 2

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    @pytest.mark.parametrize('async_degree', [2, 4])
    def test_basic(self, async_degree):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)

        eqn = Eq(u.forward, u + 1)

        op0 = Operator(eqn, opt='noop')
        op1 = Operator(eqn, opt=('buffering', 'tasking',
                                 {'buf-async-degree': async_degree}))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 3
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1
        b = buffers.pop()
        assert b.symbolic_shape[0] == async_degree

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1)

        assert np.all(u.data == u1.data)

    def test_two_heterogeneous_buffers(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)
        v1 = TimeFunction(name='v', grid=grid, save=nt)

        eqns = [Eq(u.forward, u + v + 1),
                Eq(v.forward, u + v + v.backward)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt=('buffering', 'tasking'))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 6
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 2

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_unread_buffered_function(self):
        nt = 10
        grid = Grid(shape=(4, 4))
        time = grid.time_dim

        u = TimeFunction(name='u', grid=grid, save=nt)
        u1 = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid)
        v1 = TimeFunction(name='v', grid=grid)

        eqns = [Eq(v.forward, v + 1, implicit_dims=time),
                Eq(u, v)]

        op0 = Operator(eqns, opt='noop')
        op1 = Operator(eqns, opt=('buffering', 'tasking'))

        # Check generated code
        assert len(retrieve_iteration_tree(op1)) == 2
        buffers = [i for i in FindSymbols().visit(op1) if i.is_Array]
        assert len(buffers) == 1

        op0.apply(time_M=nt-2)
        op1.apply(time_M=nt-2, u=u1, v=v1)

        assert np.all(u.data == u1.data)
        assert np.all(v.data == v1.data)

    def test_tasking_in_isolation(self):
        nt = 10

        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)

        tmp = Function(name='tmp', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=nt)

        eqns = [Eq(tmp, u + 1),
                Eq(u.forward, tmp, subdomain=bundle0)]

        op = Operator(eqns, opt=('tasking', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op)) == 2
        locks = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(locks) == 1
        assert type(locks.pop()).__base__ == Lock
        # Check waits and set-locks are at the right depth
        sections = FindNodes(Section).visit(op)
        assert len(sections) == 2
        assert str(sections[0].body[0].body[0].body[0]) == 'while(lock0[0] == 0);'
        assert (str(sections[1].body[0].body[0]) ==
                'if (thread0.joinable())\n{\n  thread0.join();\n}')
        assert str(sections[1].body[0].body[1].body[0]) == 'lock0[0] = 0;'
        assert sections[1].body[0].body[1].body[1].body[0].is_Call

        op.apply(time_M=nt-2)

        assert np.all(u.data[nt-1] == 8)

    def test_tasking_fused(self):
        nt = 10

        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)

        tmp = Function(name='tmp', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)

        eqns = [Eq(tmp, u + 1),
                Eq(u.forward, tmp, subdomain=bundle0),
                Eq(v.forward, tmp, subdomain=bundle0)]

        op = Operator(eqns, opt=('tasking', 'fuse', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op)) == 2
        locks = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(locks) == 1  # Only 1 because it's only `tmp` that needs protection
        assert type(locks.pop()).__base__ == Lock
        assert len(op._func_table) == 1
        exprs = FindNodes(Expression).visit(op._func_table['copy_device_to_host0'].root)
        assert len(exprs) == 3
        assert str(exprs[0]) == 'lock0[0] = 1;'
        assert exprs[1].write is u
        assert exprs[2].write is v

        op.apply(time_M=nt-2)

        assert np.all(u.data[nt-1] == 8)
        assert np.all(v.data[nt-1] == 8)

    def test_tasking_fused_two_locks(self):
        nt = 10

        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)

        tmp0 = Function(name='tmp0', grid=grid)
        tmp1 = Function(name='tmp1', grid=grid)
        u = TimeFunction(name='u', grid=grid, save=nt)
        v = TimeFunction(name='v', grid=grid, save=nt)

        eqns = [Eq(tmp0, u + 1),
                Eq(tmp1, v + 1),
                Eq(u.forward, tmp0, subdomain=bundle0),
                Eq(v.forward, tmp1, subdomain=bundle0)]

        op = Operator(eqns, opt=('tasking', 'fuse', 'orchestrate'))

        # Check generated code
        assert len(retrieve_iteration_tree(op)) == 2
        locks = [i for i in FindSymbols().visit(op) if i.is_Array]
        assert len(locks) == 2
        assert type(locks.pop()).__base__ == Lock
        assert type(locks.pop()).__base__ == Lock
        # Check waits and set-locks are at the right depth
        sections = FindNodes(Section).visit(op)
        assert len(sections) == 2
        assert (str(sections[0].body[0].body[0].body[0]) ==
                'while(lock0[0] == 0 && lock1[0] == 0);')  # Wait-lock
        assert (str(sections[1].body[0].body[0]) ==
                'if (thread0.joinable())\n{\n  thread0.join();\n}')  # Wait-thread
        assert str(sections[1].body[0].body[1].body[0]) == 'lock0[0] = 0;'  # Set-lock
        assert str(sections[1].body[0].body[1].body[1]) == 'lock1[0] = 0;'  # Set-lock
        assert sections[1].body[0].body[1].body[2].body[0].is_Call
        assert len(op._func_table) == 1
        exprs = FindNodes(Expression).visit(op._func_table['copy_device_to_host0'].root)
        assert len(exprs) == 4
        assert str(exprs[0]) == 'lock0[0] = 1;'
        assert str(exprs[1]) == 'lock1[0] = 1;'
        assert exprs[2].write is u
        assert exprs[3].write is v

        op.apply(time_M=nt-2)

        assert np.all(u.data[nt-1] == 8)
        assert np.all(v.data[nt-1] == 8)

    def test_attempt_tasking_but_no_temporaries(self):
        grid = Grid(shape=(10, 10, 10))

        u = TimeFunction(name='u', grid=grid, save=10)

        op = Operator(Eq(u.forward, u + 1), opt=('tasking', 'orchestrate'))

        # Degenerates to host execution with no data movement, since `u` is
        # a host Function
        piters = FindNodes(OpenMPIteration).visit(op)
        assert len(piters) == 1
        assert type(piters.pop()) == OpenMPIteration

    def test_tasking_multi_output(self):
        nt = 10

        bundle0 = Bundle()
        grid = Grid(shape=(10, 10, 10), subdomains=bundle0)
        t = grid.stepping_dim
        x, y, z = grid.dimensions

        u = TimeFunction(name='u', grid=grid, time_order=2)
        usave = TimeFunction(name='usave', grid=grid, save=nt)

        eqns = [Eq(u.forward, u + 1),
                Eq(usave, u.forward + u + u.backward + u[t, x-1, y, z],
                   subdomain=bundle0)]

        op = Operator(eqns, opt=('tasking', 'orchestrate'))
        from IPython import embed; embed()

    def test_fetching_simple(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=nt)

        for i in range(nt):
            usave.data[i, :] = i

        eqn = Eq(f, f + usave)

        op = Operator(eqn, opt='fetching')  #TODO: change names in code

        # Check generated code
        from IPython import embed; embed()

        op.apply(time_M=nt-2)

        assert np.all(f.data == 36)

    def test_fetching_two_buffers(self):
        nt = 10
        grid = Grid(shape=(4, 4))

        f = Function(name='f', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, save=nt)
        vsave = TimeFunction(name='vsave', grid=grid, save=nt)

        for i in range(nt):
            usave.data[i, :] = i
            vsave.data[i, :] = i

        eqn = Eq(f, f + usave + vsave)

        op = Operator(eqn, opt='fetching')  #TODO: change names in code
        from IPython import embed; embed()

        op.apply(time_M=nt-2)

        assert np.all(f.data == 72)

    # Below more real-life-like tests

    @pytest.mark.parametrize('opt', [('tasking',), ('buffering', 'tasking')])
    @pytest.mark.parametrize('gpu_fit', [True, False])
    def test_save(self, opt, gpu_fit):
        nt = 10
        grid = Grid(shape=(300, 300, 300))

        time_dim = grid.time_dim

        factor = Constant(name='factor', value=2, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)
        # For the given `nt` and grid shape, `usave` is roughly 4*5*300**3=~ .5GB of data

        op = Operator([Eq(u.forward, u + 1), Eq(usave, u.forward)],
                      opt=(opt, {'gpu-fit': usave if gpu_fit else None}))

        op.apply(time_M=nt-1)

        assert all(np.all(usave.data[i] == 2*i + 1) for i in range(usave.save))

    @pytest.mark.parametrize('gpu_fit', [True, False])
    def test_xcorlike_from_saved(self, gpu_fit):
        nt = 10

        grid = Grid(shape=(10, 10, 10))
        time_dim = grid.time_dim

        period = 2
        factor = Constant(name='factor', value=period, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        g = Function(name='g', grid=grid)
        v = TimeFunction(name='v', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)

        for i in range(int(nt//period)):
            usave.data[i, :] = i
        v.data[:] = i*2 + 1

        # Assuming nt//period=5, we are computing, over 5 iterations:
        # g = 4*4  [time=8] + 3*3 [time=6] + 2*2 [time=4] + 1*1 [time=2]
        op = Operator([Eq(v.backward, v - 1), Inc(g, usave*(v/2))],
                      platform='nvidiaX', language='openacc',
                      opt=('advanced', {'gpu-fit': usave if gpu_fit else None}))

        op.apply(time_M=nt-1)

        assert np.all(g.data == 30)
