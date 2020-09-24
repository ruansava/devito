import pytest
import numpy as np

from conftest import skipif
from devito import (Grid, Constant, Function, TimeFunction, Eq, Operator,
                    ConditionalDimension, norm, solve)
from devito.data import LEFT
from devito.ir.iet import FindNodes, retrieve_iteration_tree
from devito.passes.iet import OpenMPIteration
from devito.core.gpu_openmp import DeviceOpenMPIteration
from examples.seismic import TimeAxis, RickerSource, Receiver


class TestCodeGeneration(object):

    def test_basic(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid)

        op = Operator(Eq(u.forward, u + 1), platform='nvidiaX', language='openacc')

        trees = retrieve_iteration_tree(op)
        assert len(trees) == 1

        assert trees[0][1].pragmas[0].value ==\
            'acc parallel loop collapse(3)'
        assert op.body[1].header[0].value ==\
            ('acc enter data copyin(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert str(op.body[1].footer[0]) == ''
        assert op.body[1].footer[1].contents[0].value ==\
            ('acc exit data copyout(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[1].footer[1].contents[1].value ==\
            ('acc exit data delete(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')

    def test_save(self):
        nt = 10

        grid = Grid(shape=(10, 10, 10))
        time_dim = grid.time_dim

        factor = Constant(name='factor', value=2, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)

        eqns = [Eq(u.forward, u + 1), Eq(usave, u.forward)]

        # Make sure `u[t1]` is copied back to the host
        op = Operator(eqns, platform='nvidiaX', language='openacc')
        from IPython import embed; embed()
        assert len(op.body[1].header) == 2
        assert op.body[1].header[0].value ==\
            ('acc enter data copyin(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert str(op.body[1].header[1]) == ''
        assert 'copyin(usave' not in str(op)
        trees = retrieve_iteration_tree(op)
        assert len(trees) == 2
        hpis = FindNodes(OpenMPIteration).visit(op)
        #TODO: Filter out DeviceOpenMPIteration
        assert len(hpis) == 1
        hpi = hpis[0]
        assert len(hpi.header) == 1
        assert 'acc update self(u[t' in hpi.header[0].value
        assert '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])'\
            in hpi.header[0].value

        # Now telling the compiler to keep `usave` on the GPU until the very end
        op = Operator(eqns, opt=('advanced', {'device-fit': usave}),
                      platform='nvidiaX', language='openacc')
        assert len(op.body[1].header) == 3
        assert op.body[1].header[0].value ==\
            ('acc enter data copyin(u[0:u_vec->size[0]]'
             '[0:u_vec->size[1]][0:u_vec->size[2]][0:u_vec->size[3]])')
        assert op.body[1].header[1].value ==\
            ('acc enter data copyin(usave[0:usave_vec->size[0]]'
             '[0:usave_vec->size[1]][0:usave_vec->size[2]][0:usave_vec->size[3]])')
        assert str(op.body[1].header[2]) == ''
        hpis = FindNodes(OpenMPIteration).visit(op)
        #TODO: Filter out DeviceOpenMPIteration
        assert len(hpis) == 0


class TestOperator(object):

    @skipif('nodevice')
    def test_op_apply(self):
        grid = Grid(shape=(3, 3, 3))

        u = TimeFunction(name='u', grid=grid, dtype=np.int32)

        op = Operator(Eq(u.forward, u + 1))

        # Make sure we've indeed generated OpenACC code
        assert 'acc parallel' in str(op)

        time_steps = 1000
        op.apply(time_M=time_steps)

        assert np.all(np.array(u.data[0, :, :, :]) == time_steps)

    @skipif('nodevice')
    def test_save(self):
        nt = 10

        grid = Grid(shape=(10, 10, 10))
        time_dim = grid.time_dim

        factor = Constant(name='factor', value=2, dtype=np.int32)
        time_sub = ConditionalDimension(name="time_sub", parent=time_dim, factor=factor)

        u = TimeFunction(name='u', grid=grid)
        usave = TimeFunction(name='usave', grid=grid, time_order=0,
                             save=int(nt//factor.data), time_dim=time_sub)

        op = Operator([Eq(u.forward, u + 1), Eq(usave, u.forward)],
                      platform='nvidiaX', language='openacc')

        op.apply(time_M=nt-1)

        assert all(np.all(usave.data[i] == 2*i + 1) for i in range(usave.save))

    @skipif('nodevice')
    def test_iso_ac(self):
        shape = (101, 101)
        extent = (1000, 1000)
        origin = (0., 0.)

        v = np.empty(shape, dtype=np.float32)
        v[:, :51] = 1.5
        v[:, 51:] = 2.5

        grid = Grid(shape=shape, extent=extent, origin=origin)

        t0 = 0.
        tn = 1000.
        dt = 1.6
        time_range = TimeAxis(start=t0, stop=tn, step=dt)

        f0 = 0.010
        src = RickerSource(name='src', grid=grid, f0=f0,
                           npoint=1, time_range=time_range)

        domain_size = np.array(extent)

        src.coordinates.data[0, :] = domain_size*.5
        src.coordinates.data[0, -1] = 20.

        rec = Receiver(name='rec', grid=grid, npoint=101, time_range=time_range)
        rec.coordinates.data[:, 0] = np.linspace(0, domain_size[0], num=101)
        rec.coordinates.data[:, 1] = 20.

        u = TimeFunction(name="u", grid=grid, time_order=2, space_order=2)
        m = Function(name='m', grid=grid)
        m.data[:] = 1./(v*v)

        pde = m * u.dt2 - u.laplace
        stencil = Eq(u.forward, solve(pde, u.forward))

        src_term = src.inject(field=u.forward, expr=src * dt**2 / m)
        rec_term = rec.interpolate(expr=u.forward)

        op = Operator([stencil] + src_term + rec_term)

        # Make sure we've indeed generated OpenACC code
        assert 'acc parallel' in str(op)

        op(time=time_range.num-1, dt=dt)

        assert np.isclose(norm(rec), 490.56, atol=1e-2, rtol=0)


class TestMPI(object):

    @skipif('nodevice')
    @pytest.mark.parallel(mode=2)
    def test_basic(self):
        grid = Grid(shape=(6, 6))
        x, y = grid.dimensions
        t = grid.stepping_dim

        u = TimeFunction(name='u', grid=grid, space_order=2)
        u.data[:] = 1.

        expr = u[t, x, y-1] + u[t, x-1, y] + u[t, x, y] + u[t, x, y+1] + u[t, x+1, y]
        op = Operator(Eq(u.forward, expr), platform='nvidiaX', language='openacc')

        # Make sure we've indeed generated OpenACC+MPI code
        assert 'acc parallel' in str(op)
        assert len(op._func_table) == 4

        op(time_M=1)

        glb_pos_map = grid.distributor.glb_pos_map
        if LEFT in glb_pos_map[x]:
            assert np.all(u.data[0] == [[11., 16., 17., 17., 16., 11.],
                                        [16., 23., 24., 24., 23., 16.],
                                        [17., 24., 25., 25., 24., 17.]])
        else:
            assert np.all(u.data[0] == [[17., 24., 25., 25., 24., 17.],
                                        [16., 23., 24., 24., 23., 16.],
                                        [11., 16., 17., 17., 16., 11.]])

    @skipif('nodevice')
    @pytest.mark.parallel(mode=2)
    def test_iso_ac(self):
        TestOperator().test_iso_ac()
