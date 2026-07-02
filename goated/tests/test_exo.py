
from pathlib import Path

import numpy as np
import pytest
import netCDF4 as nc
import zipfile
from goated.utils.exo import ExoInfo


def make_synthetic_exodus(tmp_path):
    """
    Create a tiny Exodus-II–style netCDF4 file in tmp_path / "test.nc" with:

      - 4 nodes at (0,0,0),(1,0,0),(0,1,0),(1,1,0)
      - one quad element
      - 2 nodal variables
      - 3 timesteps

    Returns
    -------
    filename : str
        Full pathname of the created file.
    """
    # output path
    fn = tmp_path / "test.nc"
    ds = nc.Dataset(str(fn), 'w')

    # dimensions
    nvar = 2
    nnode = 4
    ntime = 3
    nstr = 16
    ds.createDimension('num_nod_var', nvar)
    ds.createDimension('num_nodes',  nnode)
    ds.createDimension('time_step',  ntime)
    ds.createDimension('strlen',     nstr)
    ds.createDimension('num_elem',   1)
    ds.createDimension('num_nodes_per_elem', 4)

    # node coordinates
    coords = np.array([[0,0,0],
                       [1,0,0],
                       [0,1,0],
                       [1,1,0]], dtype=float)
    ds.createVariable('coordx','f8',('num_nodes',))[:] = coords[:,0]
    ds.createVariable('coordy','f8',('num_nodes',))[:] = coords[:,1]
    ds.createVariable('coordz','f8',('num_nodes',))[:] = coords[:,2]

    # time_whole
    ds.createVariable('time_whole','f8',('time_step',))[:] = np.linspace(0,1,ntime)

    # 2 nodal variable fields: shape (ntime, num_nodes)
    for v in range(nvar):
        name = f'vals_nod_var{v+1}'
        var = ds.createVariable(name, 'f8', ('time_step','num_nodes'))
        # fill so that at (t,n) the value = t + v
        arr = np.arange(ntime,dtype=float)[:,None] + float(v)
        var[:] = np.tile(arr, (1,nnode))

    # name_nod_var (S1 array of shape (nvar, strlen))
    name_arr = np.empty((nvar,nstr), 'S1')
    for v in range(nvar):
        s = f"v{v+1}"
        # pad/truncate to strlen
        bs = s.encode('ascii')[:nstr].ljust(nstr,b'\0')
        name_arr[v,:] = np.frombuffer(bs, dtype='S1')
    nm = ds.createVariable('name_nod_var','S1',('num_nod_var','strlen'))
    nm[:] = name_arr

    # single quad element (1-based connectivity)
    conn = np.array([[1,2,4,3]], dtype=np.int32)
    cn = ds.createVariable('connect1','i4',('num_elem','num_nodes_per_elem'))
    cn[:] = conn

    ds.close()
    return str(fn)


def test_read_and_slice(tmp_path):
    fn = make_synthetic_exodus(tmp_path)
    exo = ExoInfo()
    exo.read_sheet(fn, z_slice=0.0)

    # we sliced at z=0 so only one z‐value
    assert np.allclose(exo.z, [0.0])
    # x and y must be {0,1}
    assert set(exo.x.tolist()) == {0.0,1.0}
    assert set(exo.y.tolist()) == {0.0,1.0}

    # tensor_data.shape = (nx,ny,nvar,ntime) = (2,2,2,3)
    assert exo.tensor_data.shape == (2,2,2,3)
    assert exo.var_mode == 2

    # node_ind has one row per original node
    assert exo.node_ind.shape == (4,3)

    # elem_ind: zero-based version of [[1,2,4,3]]
    assert exo.elem_ind.shape == (1,4)
    np.testing.assert_array_equal(exo.elem_ind, np.array([[0,1,3,2]]))


def test_read_sheet_from_zip(tmp_path):
    fn = make_synthetic_exodus(tmp_path)
    archive_path = f"{fn}.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.write(fn, arcname=fn.split("/")[-1])

    # remove the uncompressed file so the loader must fall back to the zip archive
    import os
    os.remove(fn)

    exo = ExoInfo()
    exo.read_sheet(fn, z_slice=0.0)

    assert np.allclose(exo.z, [0.0])
    assert exo.tensor_data.shape == (2, 2, 2, 3)


def test_read_sheet_accepts_path_object(tmp_path):
    fn = make_synthetic_exodus(tmp_path)
    exo = ExoInfo()
    exo.read_sheet(Path(fn), z_slice=0.0)

    assert np.allclose(exo.z, [0.0])
    assert exo.tensor_data.shape == (2, 2, 2, 3)


def test_compute_det_jac():
    exo = ExoInfo()
    # unit-square nodes at (0,0),(1,0),(0,1),(1,1)
    x = np.array([0,1,0,1],float)
    y = np.array([0,0,1,1],float)
    z = np.zeros(4)
    # pick reference quad center
    gp = np.array([0.0,0.0])
    J = exo.compute_det_jac(x,y,z,gp)
    # for bilinear mapping of unit square, det(J) = area / 4 = 1/4
    assert pytest.approx(J, rel=1e-8) == 0.25


def test_compute_spatial_integral(tmp_path):
    fn = make_synthetic_exodus(tmp_path)
    exo = ExoInfo()
    exo.read_sheet(fn, z_slice=0.0)

    X = exo.tensor_data  # shape (2,2,2,3)
    # var indices 0,1; time indices 0,1,2
    var  = [0,1]
    time = [0,1,2]

    # func(v) = v[:,:,0,:] + v[:,:,1,:]  ⇒  (t+0)+(t+1)=2t+1
    func  = lambda v: np.sum(v, axis=2)
    # deriv = ones
    deriv = lambda v: np.ones_like(v)

    # compute only function
    p0, j0 = exo.compute_spatial_integral(
        X, var, time,
        func=func, deriv=deriv,
        compute_func=True, compute_deriv=False
    )
    # jacobian must be empty
    assert isinstance(j0, np.ndarray) and j0.size == 0

    # now compute derivative too
    p, jac = exo.compute_spatial_integral(
        X, var, time,
        func=func, deriv=deriv,
        compute_func=True, compute_deriv=True
    )
    # with detJ=1/4 and 4 gp's per element, p[t] = (2t+1)*area = (2t+1)
    expected_p = np.array([1.0, 3.0, 5.0])
    assert np.allclose(p, expected_p)

    # jac shape matches X
    assert jac.shape == X.shape
    # and must not be all zeros
    assert np.any(jac != 0.0)


def test_write_sheet(tmp_path):
    fn = make_synthetic_exodus(tmp_path)
    exo = ExoInfo()
    exo.read_sheet(fn, z_slice=0.0)

    # make a trivial modification
    X = exo.tensor_data + 10.0

    out_fn = tmp_path / "out.nc"
    exo.write_sheet(X, vars=[0,1], fname=str(out_fn), template_fname=fn)

    # read it back
    exo2 = ExoInfo()
    exo2.read_sheet(str(out_fn), z_slice=0.0)

    # the tensor_data must round‐trip
    assert np.allclose(exo2.tensor_data, X)
