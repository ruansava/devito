import pytest
import numpy as np

from devito import Eq, Grid, Function, SubDomain, TimeFunction, Operator
from devito.archinfo import get_gpu_info
from devito.ir import FindNodes, FindSymbols, Section, retrieve_iteration_tree
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

        # We use a subdomain to enforce Eqs to end up in different loops
        class Bundle(SubDomain):
            name = 'bundle'

            def define(self, dimensions):
                x, y, z = dimensions
                return {x: ('middle', 0, 0), y: ('middle', 0, 0), z: ('middle', 0, 0)}

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
        assert (str(sections[0].body[0].body[0].body[0]) ==
                'while(lock0[0] == 0);')  # Wait-lock
        assert (str(sections[1].body[0].body[0]) ==
                'if (thread0.joinable())\n{\n  thread0.join();\n}')  # Wait-thread
        assert (str(sections[1].body[0].body[1].body[0]) ==
                'lock0[0] = 0;')  # Set-lock
        assert sections[1].body[0].body[1].body[1].body[0].is_Call

        op.apply(time_M=nt-2)

        assert np.all(u.data[nt-1] == 8)

    @pytest.mark.parametrize('gpu_fit', [True, False])
    def test_save(self, gpu_fit):
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
                      opt=('advanced', {'gpu-fit': usave if gpu_fit else None}))

        op.apply(time_M=nt-1)

        assert all(np.all(usave.data[i] == 2*i + 1) for i in range(usave.save))
