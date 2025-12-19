import numpy as np
from typing import Tuple
from pyttb import tensor, ttensor, ktensor  # type: ignore
from typing import TypeAlias, Tuple, List, Optional, Sequence

Tensor : TypeAlias = tensor | ttensor | ktensor


class Goal:
    """
    Let G = Goal(ground_truth, **kwargs). For a Tensor U and a
    boolean flag, let

        (*) s          = G.computeScalar(U),       and
        (*) (vec, jac) = G.computeVector(U, flag), and
        (*) dsdU       = G.computeGrad(U).
    
    These quantities have the following properties.

        For a Tensor V, val(V) = s + dot(dsdU, V - U) is the first
        order approximation of G.computeScalar(V) at U.

        The identity s == np.linalg.norm(vec - G.target, 'fro')**2
        always holds.

        If flag is True, then jac.shape == G.domain_shape, and jac
        is an array that can be used in certain ways to compute
        vector products with the Hessian of G.computeScalar(U).
    """

    def __init__(self, ground_truth : Tensor, **kwargs) -> None:
        self.domain_shape = ground_truth.shape
        self.cached_vec  : np.ndarray = np.empty(())
        self.cached_jac  : np.ndarray = np.empty(())
        self.target, _ = self.computeVector(ground_truth, compute_deriv=False)
        return
    
    def computeVector(self, U : Tensor, compute_deriv=False) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def computeScalar(self, U : Tensor) -> np.floating:
        vec, _ = self.computeVector(U, compute_deriv=False)
        diff = vec - self.target
        F = np.linalg.norm(diff)**2
        return F

    def computeGrad(self, U : Tensor) -> np.ndarray:
        raise NotImplementedError()


class PhysicsGoal(Goal):

    def __init__(self, ground_truth : Tensor, var, time):
        if not isinstance(var, np.ndarray):
            var = np.array(var)
        if not isinstance(time, np.ndarray):
            time = np.array(time)
        self.var  : np.ndarray = var
        self.time : np.ndarray = time
        super().__init__(ground_truth)
        jac_m, jac_n = self.domain_shape[:2]
        i1 = np.arange(jac_m)  # arange of domain_shape[0]
        i2 = np.arange(jac_n)  # arange of domain_shape[1]
        i3 = self.var          # subset of arange domain_shape[2]
        i4 = self.time         # subset of arange domain_shape[3]
        self._nonconst_indices : tuple[np.ndarray,...] = np.ix_(i1, i2, i3, i4)
        self.DEBUG = False

    def grad_from_vec_and_jac(self, v, j):
        diff = v - self.target
        grad = j.copy()
        grad[self._nonconst_indices] *= 2*np.reshape(diff, (1,1,1) + self.target.shape)
        # Question: can I do away with the self._grad_indices, and just use
        #   broadcast_dims = (1,)*(jac.ndim - self.target.ndim)
        #   jac[:] *= 2*np.reshape(diff, broadcast_dims + self.target.shape)
        # ?
        return grad
    
    def computeGrad(self, U : Tensor, use_cached: bool):
        if use_cached:
            vec, jac = self.cached_vec, self.cached_jac
        else:
            vec, jac = self.computeVector(U, compute_deriv=True)
            # ^ It seems weird that vec.size is a different shape than jac.shape[-1].
            #   ... what is the derivative with respect to?
        return self.grad_from_vec_and_jac(vec, jac)

    def _gn_hessvec(self, Md) -> np.ndarray:
        i3 = self.var
        i4 = self.time
        jac = self.cached_jac
        jact = jac[ :, :, :, i4 ]
        Mdt  =  Md[ :, :, :, i4 ]
        val_dot = np.zeros((i4.size,))
        val_dot[:] = np.einsum('hijk,hijk->k', jact[:,:,i3,:], Mdt[:,:,i3,:])
        if self.DEBUG:
            vd = np.zeros((i4.size,))
            for i in range(i4.size):
                vd[i] = np.reshape(jac[:,:,i3,i4[i]],(1,-1),order='F') @ np.reshape(Md[:,:,i3,i4[i]],(-1,1),order='F')
            assert np.linalg.norm(vd - val_dot) <= 1e-8 * np.maximum(1.0, np.linalg.norm(vd))

        # compute dot gradient tensor dF/dM(i,j,v,t)
        jac_dot = np.zeros(jac.shape)
        mask_t  = np.zeros(jac.shape, dtype=bool)
        mask_v  = np.zeros(jac.shape, dtype=bool)
        mask_t[:,:,:,i4] = True
        mask_v[:,:,i3,:]  = True 
        ro = 'C'  # doesn't matter, just be explicit.
        mask = (mask_t & mask_v).ravel(order=ro)
        jac_dot.ravel(order=ro)[mask] = (2*val_dot[None,None,:]*jact[:,:,i3,:]).ravel(order=ro)
        if self.DEBUG:
            jd = np.zeros(jac.shape)
            for i in range(i4.size):
                jd[:,:,i3,i4[i]] = 2*val_dot[i]*jac[:,:,i3,i4[i]]
            assert np.linalg.norm(jac_dot - jd) <= 1e-8 * np.maximum(1.0, np.linalg.norm(jd))
        return jac_dot

    def _tucker_gn_block_diag_goal_updates(self, w: float, M: ttensor, scaler, factor_cols: list[list]):
        scale = np.sqrt(2 * w)
        jac   = self.cached_jac
        shape = self.domain_shape
        jac_mat_shape = shape[:-1] + (1,)
        ndim = len(shape)
        i4 = self.time
        for t in i4:
            jac_t = tensor(jac[:,:,:,t], shape=jac_mat_shape, copy=False)
            jac_t = scaler.unscale_tensor(jac_t, shift=False)
            for n in range(ndim):
                mats, dims = [], []
                for i in range(ndim):
                    if i != n:
                        if i == ndim-1:
                            mats.append(np.reshape(M.factor_matrices[i][t,:],(1,-1)))
                        else:
                            mats.append(M.factor_matrices[i])
                        dims.append(i)
                D = jac_t.ttm(mats, dims=dims, transpose=True)
                D = D.to_tenmat(np.array([n])).data @ M.core.to_tenmat(np.array([n])).data.T
                if n == ndim-1:
                    D2 = np.zeros((shape[n], M.core.shape[n]))
                    D2[t,:] = D
                    D = D2
                D = scale * np.reshape(D, (-1,1), order='F')
                factor_cols[n].append(D)

            # Core term
            mats = [M.factor_matrices[i] for i in range(ndim-1)]
            mats.append(np.reshape(M.factor_matrices[-1][t,:],(1,-1)))
            dims = list(range(ndim))
            D = jac_t.ttm(mats, dims=dims, transpose=True)
            D = scale * np.reshape(D.data, (-1,1), order='F')
            factor_cols[-1].append(D)
        return


class Goals:
    """
    Constituent PhysicsGoal objects define their targets in terms 
    of the un-scaled tensor, but GocchaObjective works in terms of
    a scaled tensor. So this class maintains its own scaler.
    """

    def __init__(self, scaler, goals : List[PhysicsGoal], weights : Optional[Sequence[float] | np.ndarray] = None):
        assert len(goals) > 0
        self.scaler = scaler
        self.goals = goals
        if weights is None:
            weights = np.ones((len(goals),), dtype=float)
        elif not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        self.weights : np.ndarray = weights
        self.M = None
        self.Mfs : tensor  = tensor()
        self._shape : Tuple[int,...] = goals[0].domain_shape
        self._ndim  : int = len(self._shape)
        self._grad  : tensor = tensor()
        return
        
    def update(self, M, Mf: tensor, grad=True, jacs=True):
        self.M = M  # not used in this base class
        self.Mfs = self.scaler.unscale_tensor(Mf)
        if grad or jacs:
            self.recompute_jacs()
        if grad:
            self.recompute_grad(use_cached_jacs=True)
        return

    def recompute_jacs(self) -> None:
        for _,g in zip(self.weights,self.goals):
            val, jac = g.computeVector(self.Mfs, compute_deriv=True)
            g.cached_vec = val
            g.cached_jac = jac
        return

    def value(self) -> float:
        F = 0
        for w,g in zip(self.weights, self.goals):
            F += w * g.computeScalar(self.Mfs)
        return F

    def recompute_grad(self, use_cached_jacs: bool) -> None:
        Yg = np.zeros(self._shape)
        for w,g in zip(self.weights, self.goals):
            Yg += w * g.computeGrad(self.Mfs, use_cached_jacs)
        Yg = tensor(Yg)
        Yg = self.scaler.unscale_tensor(Yg, shift=False)
        self._grad : tensor = Yg
        return

    def gradient_wrt_reconstruction(self):
        return self._grad

    def hessvec_wrt_reconstruction(self, Md: tensor) -> tensor:
        # Md = self.scaler.unscale_tensor(Md, shift=False).data
        Yd = np.zeros(self._shape, order='F')
        for w,g in zip(self.weights, self.goals):
            jac_dot = g._gn_hessvec(Md)
            Yd += w*jac_dot
        Yd = self.scaler.unscale_tensor(Yd, shift=False)
        return Yd

    def eval_goals(self, U: tensor, scaled=False):   
        if scaled:
            U = self.scaler.unscale_tensor(U)
        v = np.array([g.computeScalar(U) for g in self.goals])
        return v 

    def auto_reweight(self, U: tensor, scaled=False):
        v = self.eval_goals(U, scaled=scaled)
        self.weights = 1 / (v * len(self.goals))
        return


class CPGoals(Goals):

    def __init__(self, scaler, goals: List[PhysicsGoal], weights: Sequence[float] | np.ndarray | None = None):
        super().__init__(scaler, goals, weights)
        self.M : ktensor = ktensor()

    def update(self, M: ktensor, Mf: tensor, grad=True, jacs=True):
        super().update(M, Mf, grad, jacs)
        return


class TuckerGoals(Goals):

    def __init__(self, scaler, goals: List[PhysicsGoal], weights: Sequence[float] | np.ndarray | None = None):
        super().__init__(scaler, goals, weights)
        self.M : ttensor = ttensor()
        return
        
    def update(self, M: ttensor, Mf: tensor, grad=True, jacs=True):
        super().update(M, Mf, grad, jacs)
        return

    def tucker_gn_diag_block_goal_updates(self) -> list[np.ndarray]:
        factor_cols = [[] for _ in range(self._ndim + 1)]
        for w, g in zip(self.weights, self.goals):
            g._tucker_gn_block_diag_goal_updates(w, self.M, self.scaler, factor_cols)
        factors = [np.column_stack(cols) for cols in factor_cols]
        return factors
