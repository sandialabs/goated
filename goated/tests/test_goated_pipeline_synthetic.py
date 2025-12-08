import numpy as np
import pyttb as ttb
import pyrol

from goated.utils.exo import ExoInfo
from goated.goals.physics import MomentumGoal
from goated.tucker import TuckerObjective
from goated.goals.abstract import Goal
from goated.tucker import TuckerGoals
from goated.rol_interface.models import GoatedRolModel
from goated.utils import scalers as sc


class DummyExo(ExoInfo):
    """
    A fake ExoInfo that never reads from disk but will interpret
    a tensor X of shape (nx,ny,nvar,ntime) as if it were nodal data,
    and defines compute_spatial_integral to be a simple sum-over-space.
    """
    def __init__(self, nx, ny):
        # we just need x,y,z,t coords and node maps
        self.x = np.arange(nx, dtype=float)
        self.y = np.arange(ny, dtype=float)
        self.z = np.array([0.0])
        # build a trivial mesh of nx*ny nodes in lexicographic order
        xs, ys = np.meshgrid(self.x, self.y, indexing='ij')
        self.x_coord = xs.ravel()
        self.y_coord = ys.ravel()
        self.z_coord = np.zeros_like(self.x_coord)
        # there is exactly one distinct z-slice
        self.node_ind = np.vstack((xs.ravel().astype(int),
                                   ys.ravel().astype(int),
                                   np.zeros_like(xs.ravel(),dtype=int)
                                   )).T
        self.tensor_node_ind = self.node_ind[:,0:2]
        # one dummy quad per 4 nodes (not really used in our sum)
        self.elem_ind = np.arange(self.x_coord.size).reshape(1,-1)
        # trivial integration tables
        self.w_det_J = np.ones((1,1))
        # we need A, gp, linear_elem_ind, node_linear_ind, tensor_elem_linear_ind
        # but the compute_spatial_integral below will ignore them
        # so we can simply stub them out:
        self.A = np.eye(1)
        self.gp = np.zeros((1,2))
        self.linear_elem_ind = np.arange(nx*ny)
        self.node_linear_ind = np.arange(nx*ny)
        self.tensor_elem_linear_ind = np.arange(nx*ny).reshape(1,-1)
        self.var_mode = 2

    def compute_spatial_integral(self,
                                 X,          # ndarray (nx,ny,nvar,ntime)
                                 var,        # list of int
                                 time,       # list of int
                                 func,
                                 deriv=None,
                                 compute_func=True,
                                 compute_deriv=False):
        """
        We ignore all the fancy FE machinery and simply sum func(...) over
        the two spatial dims and the var‐subset.  Jacobs are either 0 or 1.
        """
        # extract the sub‐array
        if hasattr(X, 'full'):  # handle ktensor case
            Xf = X.full().double()
        else:
            Xf = X
        # Xf shape = (nx,ny,nvar,ntime)
        sub = Xf[np.ix_(range(Xf.shape[0]),
                        range(Xf.shape[1]),
                        var,
                        time)]  # shape => (nx,ny,len(var),len(time))

        # build values at each (i,j,var,t)
        if compute_func:
            fv = func(sub)   # expects shape (nx,ny,len(time))
            # sum over i,j => produce one number per time
            p = fv.sum(axis=(0,1))
        else:
            p = np.zeros(len(time))

        if compute_deriv:
            # we assign a trivial Jacobian of all ones
            jac = np.ones_like(Xf)
        else:
            jac = np.empty((0,), dtype=float)

        return p, jac


def test_goated_pipeline_synthetic(nx=5, ny=5, nvar=5, nt=5, rank_approx=1):
    """
    Integration‐style smoke test: build a tiny synthetic 4‐way tensor,
    define a single momentum goal, run the full GoatedRolModel solver,
    and assert that the goal errors shrink.
    """
    # 1) Build a very simple “simulation” tensor X_data
    X_data = np.zeros((nx, ny, nvar, nt), dtype=float)
    for i in range(nx):
        for j in range(ny):
            for v in range(nvar):
                for t in range(nt):
                    X_data[i,j,v,t] = np.sqrt(i + 2*1/(j+1) + 3*v**0.5 + t**1.5)
    X_tt = ttb.tensor(X_data)

    # 2) Hook up our dummy ExoInfo
    exo = DummyExo(nx, ny)

    var_idx  = list(range(nvar))
    time_idx = list(range(nt))
    mg = MomentumGoal(X_tt, var_idx, time_idx, exo)
    # ^ why does MomentumGoal need X_tt ?
    scaler = sc.StdScaler(X_tt, exo.var_mode)
    goals  = TuckerGoals(scaler, [mg])

    # 5) Build the “Gotcha” objective: we’ll weight the data‐fit and goal equally
    a = 1.0
    b = 1.0
    Xs = scaler.scale_tensor(X_tt)
    from goated.tucker import TuckerObjective, GotchaObjective
    gotcha = GotchaObjective(Xs, scaler, goals, a, b)

    # 6) Get an initial Tucker decomposition (just the trivial rank‐1 HOSVD)
    #    Note: we use the built‐in ttensor.hosvd to full HOSVD and then truncate
    us0 = ttb.hosvd(Xs, tol=1e-15)
    
    # Now truncate to the desired rank:
    truncated_factors = []
    for U in us0.factor_matrices:
        truncated_factors.append(U[:, :rank_approx])
    initial_decomp = ttb.ttensor(
        us0.core[:rank_approx, :rank_approx, :rank_approx, :rank_approx], truncated_factors
    )
    before = np.array(goals.eval_goals(initial_decomp.full(), scaled=True))
    norm_before = np.linalg.norm(before)

    model = GoatedRolModel(gotcha, initial_decomp)
    params = pyrol.ParameterList()
    params['General'] =  pyrol.ParameterList()
    params['General']['Output Level'] = 1
    params['General']['Krylov'] = pyrol.ParameterList()
    params['General']['Krylov']['Iteration Limit'] = 1_000
    params['Step'] = pyrol.ParameterList()
    params['Step']['Trust Region'] = pyrol.ParameterList()
    params['Step']['Trust Region']['Initial Radius'] = 10.0
    params['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
    params['Status Test'] = pyrol.ParameterList()
    params['Status Test']['Iteration Limit'] = 20
    model.solve(rol_params=params)

    # 10) Pull out the final decomposition & reevaluate the goals
    final = model.decomp
    after = np.array(goals.eval_goals(final.full(), scaled=True))
    norm_after = np.linalg.norm(after)

    U0 = scaler.unscale_tensor(initial_decomp.full())
    fit0 = 1-(X_tt-U0).norm()/X_tt.norm()

    U1 = scaler.unscale_tensor(final.full())
    fit1 = 1-(X_tt-U1).norm()/X_tt.norm()

    assert norm_after < norm_before, (
        f"Gotcha pipeline did not reduce goal‐error: "
        f"||before||={norm_before:.6e}, ||after||={norm_after:.6e}"
    )

    pass

def test_tucker_synthetic(nx=5, ny=5, nvar=5, nt=5, rank_approx=1):
    """
    Integration‐style smoke test: build a tiny synthetic 4‐way tensor,
    define a single momentum goal, run the full GoatedRolModel solver,
    and assert that the goal errors shrink.
    """
    X_data = np.zeros((nx, ny, nvar, nt), dtype=float)
    for i in range(nx):
        for j in range(ny):
            for v in range(nvar):
                for t in range(nt):
                    X_data[i,j,v,t] = np.sqrt(i + 2*1/(j+1) + 3*v**0.5 + t**1.5)
    X_tt = ttb.tensor(X_data)
    Xs = X_tt

    gotcha = TuckerObjective(Xs)

    us0 = ttb.hosvd(Xs, tol=1e-15)
    truncated_factors = []
    for U in us0.factor_matrices:
        truncated_factors.append(U[:, :rank_approx])
    initial_decomp = ttb.ttensor(
        us0.core[:rank_approx, :rank_approx, :rank_approx, :rank_approx], truncated_factors
    )

    model = GoatedRolModel(gotcha, initial_decomp)
    params = pyrol.ParameterList()
    params['General'] =  pyrol.ParameterList()
    params['General']['Output Level'] = 1
    params['General']['Krylov'] = pyrol.ParameterList()
    params['General']['Krylov']['Iteration Limit'] = 1_000
    params['Step'] = pyrol.ParameterList()
    params['Step']['Trust Region'] = pyrol.ParameterList()
    params['Step']['Trust Region']['Initial Radius'] = 10.0
    params['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
    params['Status Test'] = pyrol.ParameterList()
    params['Status Test']['Iteration Limit'] = 20
    model.solve(rol_params=params)

    # 10) Pull out the final decomposition & reevaluate the goals
    final = model.decomp

    U0 = initial_decomp.full()
    fit0 = 1-(X_tt-U0).norm()/X_tt.norm()

    U1 = final.full()
    fit1 = 1-(X_tt-U1).norm()/X_tt.norm()

    assert fit1 <= fit0, (
        f"Tucker optimization _DECREASED_ fit; before={fit0:.6e}, after={fit1:.6e}"
    )

    pass

if __name__ == '__main__':
    #test_tucker_synthetic()
    test_goated_pipeline_synthetic()
    # test_goated_pipeline_synthetic(nx=10, ny=11, nvar=10, nt=13, rank_approx=2)
    print()
