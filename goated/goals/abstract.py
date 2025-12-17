import numpy as np
from typing import Tuple
from pyttb import tensor, ttensor, ktensor  # type: ignore
from typing import TypeAlias

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
    
    def computeGrad(self, U : Tensor):
        vec, jac = self.computeVector(U, compute_deriv=True)
        # ^ It seems weird that vec.size is a different shape than jac.shape[-1].
        #   ... what is the derivative with respect to?
        diff = vec - self.target
        grad = jac.copy()
        grad[self._nonconst_indices] *= 2*np.reshape(diff, (1,1,1) + self.target.shape)
        # Question: can I do away with the self._grad_indices, and just use
        #   broadcast_dims = (1,)*(jac.ndim - self.target.ndim)
        #   jac[:] *= 2*np.reshape(diff, broadcast_dims + self.target.shape)
        # ?
        return grad

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
