from pyttb import tensor, ttensor # type: ignore
import numpy as np

import goated.utils.linops as linops
from collections import defaultdict
from goated.goals.abstract import Goal
from typing import Tuple, List, Optional, Sequence
import time as _time


class TuckerObjective:

    def __init__(self, X, s=None):
        self.X = X
        self.s = s if s is not None else  self.X.norm()**2
        self._shape : Tuple[int,...] = X.shape
        self._ndims : int = len(self._shape)
        self.Z  : List[tensor]     = []
        self.Zt : List[np.ndarray] = []
        self.M  : ttensor = ttensor()
        self.Mf : tensor  = tensor()
        self._grad : ttensor = ttensor()
        self.times = defaultdict(list)

    def full(self, M) -> tensor:
        # Same as M.full(), except saves the intermediate tensors
        ZZ = M.core.ttm(M.factor_matrices[0],0)
        self.Z = [ ZZ ]
        for i in range(1,self._ndims):
            ZZ = ZZ.ttm(M.factor_matrices[i],i)
            self.Z.append(ZZ)
        return ZZ

    def update(self, M, prec=True, grad=True) -> None:
        self.M = M
        self.Mf = self.full(M)
        # compute intermediate tenmats used in gradient and hessvec
        self.Zt = [ M.core.to_tenmat(np.array([0])).data.T ]
        for i in range(1,self._ndims):
            self.Zt.append(self.Z[i-1].to_tenmat(np.array([i])).data.T)
        # whether we need to recompute point data in hessian/preconditioner
        if grad:
            self.recompute_grad()
        if prec:
            self.recompute_prec()
        return

    def value(self) -> float:
        Y = self.Mf - self.X
        F = (Y.norm()**2)/self.s
        return F
    
    def _deriv_wrt_reconstructed_tensor(self) -> tensor:
        Zb = (2/self.s)*(self.Mf-self.X)
        return Zb

    def recompute_grad(self) -> None:
        M = self.M
        tic = _time.time()
        Zb = self._deriv_wrt_reconstructed_tensor()
        Gf : list[np.ndarray] = [None] * self._ndims # type: ignore
        # ^ We need to reserve space for Gf since we fill it in reverse order.
        for i in reversed(range(self._ndims)):
            Gf[i] = Zb.to_tenmat(np.array([i])).data @ self.Zt[i]
            Zb = Zb.ttm(M.factor_matrices[i].T,i)
        G = ttensor(Zb, Gf)
        toc = _time.time()
        self.times['recompute_grad'].append(toc - tic)
        self._grad = G
        return

    def gradient(self) -> ttensor:
        return self._grad
    
    def _tangent_reconstructed_tensor(self, V, rescale=True) -> tensor:
        M = self.M
        Zd = M.core.ttm(V.factor_matrices[0],0) + V.core.ttm(M.factor_matrices[0],0)
        for i in range(1, self._ndims):
            Zd = self.Z[i-1].ttm(V.factor_matrices[i],i) + Zd.ttm(M.factor_matrices[i],i)
        if rescale:
            return (2/self.s)*Zd
        else:
            return Zd

    def hessvec(self, V) -> ttensor:
        M = self.M
        tic = _time.time()
        Zbd = self._tangent_reconstructed_tensor(V)
        Hv = [np.empty(())] * self._ndims
        for i in reversed(range(self._ndims)):
            Hv[i] = Zbd.to_tenmat(np.array([i])).data @ self.Zt[i]
            Zbd = Zbd.ttm(M.factor_matrices[i].T,i)
        Hv = ttensor(Zbd, Hv)
        toc = _time.time()
        self.times['gn_hessvec'].append(toc - tic)
        return Hv

    def recompute_prec(self):
        tic = _time.time()
        n = self._ndims
        M = self.M
        grams = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(n)]
        self.C = []
        for k in range(n):
            D = np.array([1])
            for i in reversed(range(n)):
                if i != k:
                    D = np.kron(D, grams[i])
            G = M.core.to_tenmat(np.array([k])).data
            D = (2/self.s) * G @ D @ G.T
            block = linops.InvPosDef(D)
            self.C.append(block)
        grams[0] *= (2/self.s)
        kron_args = [linops.InvPosDef(gram) for gram in grams[::-1]]
        B = linops.KronStructured.recursive_dyadic(kron_args)
        self.C.append(B)
        toc = _time.time()
        self.times['recompute_bd_prec'].append(toc - tic)
        return

    def precvec(self, V):        
        tic = _time.time()
        Pv = []
        for k in range(self._ndims):
            Ck = self.C[k]
            Vk = V.factor_matrices[k]
            try:
                Pvk = Vk @ Ck
            except ValueError:
                Pvk = Ck @ Vk.ravel(order='F')
                Pvk = Pvk.reshape(Vk.shape, order='F')
            Pv.append(Pvk)

        Pvc = V.core.data.ravel(order='F')
        Pvc = self.C[-1] @ Pvc
        Pvc = Pvc.reshape(V.core.shape, order='F')
        Pv  = ttensor(tensor(Pvc), Pv)
        toc = _time.time()
        self.times['gn_bd_precvec'].append(toc - tic)
        return Pv


class TuckerGoals:

    def __init__(self, scaler, goals : List[Goal], weights : Optional[Sequence[float] | np.ndarray] = None, _shape=None):
        self.scaler = scaler
        self.goals = goals
        if weights is None:
            weights = np.ones((len(self.goals),))
        self.weights = weights
        self.M   : ttensor = ttensor()
        self.Mfs : tensor  = tensor()
        if len(goals) > 0:
            self._shape : Tuple[int,...] = goals[0].domain_shape
        elif isinstance(_shape, tuple):
            self._shape : Tuple[int,...] = _shape
        else:
            raise ValueError()
        self._ndim  : int = len(self._shape)
        self._grad  : tensor = tensor()
        self._diag_hess_factors : list[np.ndarray] = []
        self.jac_times : list[float] = []
        self.DEBUG = True
        
    def update(self, M: ttensor, Mfs: tensor, grad=True, jacs: bool=True, prec=True) -> None:
        self.M = M
        self.Mfs = Mfs
        if grad or jacs or prec:
            self.recompute_jacs()
        if grad:
            self.recompute_grad()
            # ^ That needs to come after recompute_jacs().
        return
    
    def recompute_jacs(self):
        for _, g in zip(self.weights, self.goals):
            tic = _time.time()
            vec, jac = g.computeVector(self.Mfs, compute_deriv=True)
            g.cached_vec = vec
            g.cached_jac = jac
            # ^ That caches the goal's value and Jacobian for future use.
            toc = _time.time()
            self.jac_times.append(toc - tic)
        return

    def value(self):
        F = 0.0
        for w,g in zip(self.weights, self.goals):
            F += w * g.computeScalar(self.Mfs)
        return F
    
    def recompute_grad(self) -> None:
        Yg = np.zeros(self._shape)
        for w,g in zip(self.weights, self.goals):
            Yg += w * g.computeGrad(self.Mfs)
        Yg = tensor(Yg)
        Yg = self.scaler.unscale_tensor(Yg, shift=False)
        self._grad : tensor = Yg
        return

    def gradient(self) -> tensor:
        return self._grad
    
    def hessvec(self, Md: tensor) -> tensor:
        # compute necessary gradient info
        Yd = np.zeros(self._shape, order='F')
        for w,g in zip(self.weights, self.goals):
            jac  = g.cached_jac
            var  = g.var
            time = g.time

            # compute val_dot (could tangent-differentiate fcn, but since we already have jac, we just do a mat-vec)
            jact = jac[:,:,:,time]
            Mdt  = Md[:,:,:,time]
            val_dot = np.zeros((time.size,))
            val_dot[:] = np.einsum('hijk,hijk->k', jact[:,:,var,:], Mdt[:,:,var,:])
            if self.DEBUG:
                vd = np.zeros((time.size,))
                for i in range(time.size):
                    vd[i] = np.reshape(jac[:,:,var,time[i]],(1,-1),order='F') @ np.reshape(Md[:,:,var,time[i]],(-1,1),order='F')
                assert np.linalg.norm(vd - val_dot) <= 1e-8 * np.maximum(1.0, np.linalg.norm(vd))

            # compute dot gradient tensor dF/dM(i,j,v,t)
            jac_dot = np.zeros(jac.shape)
            mask_t  = np.zeros(jac.shape, dtype=bool)
            mask_v  = np.zeros(jac.shape, dtype=bool)
            mask_t[:,:,:,time] = True
            mask_v[:,:,var,:]  = True 
            ro = 'C'  # doesn't matter, just be explicit.
            mask = (mask_t & mask_v).ravel(order=ro)
            jac_dot.ravel(order=ro)[mask] = (2*val_dot[None,None,:]*jact[:,:,var,:]).ravel(order=ro)
            if self.DEBUG:
                jd = np.zeros(jac.shape)
                for i in range(time.size):
                    jd[:,:,var,time[i]] = 2*val_dot[i]*jac[:,:,var,time[i]]
                assert np.linalg.norm(jac_dot - jd) <= 1e-8 * np.maximum(1.0, np.linalg.norm(jd))

            Yd += w*jac_dot
        Yd = self.scaler.unscale_tensor(Yd, shift=False)
        return Yd

    def gn_diag_block_goal_updates(self):
        M = self.M
        factor_cols = [[] for _ in range(self._ndim + 1)]
        for w, g in zip(self.weights, self.goals):
            goal_scale = np.sqrt(2 * w)
            time = g.time
            jac  = g.cached_jac
            jac_mat_shape = self._shape[0:self._ndim-1] + (1,)
            for t in time:
                jac_t = tensor(jac[:,:,:,t], shape=jac_mat_shape, copy=False)
                jac_t = self.scaler.unscale_tensor(jac_t, shift=False)
                for n in range(self._ndim):
                    mats, dims = [], []
                    for i in range(self._ndim):
                        if i != n:
                            if i == self._ndim-1:
                                mats.append(np.reshape(M.factor_matrices[i][t,:],(1,-1)))
                            else:
                                mats.append(M.factor_matrices[i])
                            dims.append(i)
                    D = jac_t.ttm(mats, dims=dims, transpose=True)
                    D = D.to_tenmat(np.array([n])).data @ M.core.to_tenmat(np.array([n])).data.T
                    if n == self._ndim-1:
                        D2 = np.zeros((self._shape[n], M.core.shape[n]))
                        D2[t,:] = D
                        D = D2
                    D = goal_scale * np.reshape(D, (-1,1), order='F')
                    factor_cols[n].append(D)

                # Core term
                mats = [M.factor_matrices[i] for i in range(self._ndim-1)]
                mats.append(np.reshape(M.factor_matrices[-1][t,:],(1,-1)))
                dims = list(range(self._ndim))
                D = jac_t.ttm(mats, dims=dims, transpose=True)
                D = goal_scale * np.reshape(D.data, (-1,1), order='F')
                factor_cols[-1].append(D)

        factors = [np.column_stack(cols) for cols in factor_cols]
        return factors

    def eval_goals(self, U: tensor, scaled=False):   
        if scaled:
            U = self.scaler.unscale_tensor(U)
        v = np.array([g.computeScalar(U) for g in self.goals])
        return v 

    def auto_reweight(self, U: tensor, scaled=False):
        v = self.eval_goals(U, scaled=scaled)
        self.weights = 1 / (v * len(self.goals))
        return


class GotchaObjective(TuckerObjective):

    def __init__(self, X, scaler, goals : TuckerGoals, a, b, jacobi=True):
        super().__init__(X, s=1.0)
        self.scaler = scaler # Why not inherit self.scaler from goals.scaler ?
        self.goals = goals
        self.a = a
        self.b = b
        self.jacobi = jacobi
        self.block_jacobi_ops_cache = []
        self._grad : tensor = tensor()

    def update(self, M, prec=True, grad=True):
        super().update(M, grad=False, prec=False)
        self.Mfs = self.scaler.unscale_tensor(self.Mf)
        jac_times = []
        self.goals.update(self.M, self.Mfs, grad=True, jacs=True)
        self.times['recompute hessian'].extend(jac_times)
        if grad:
            self.recompute_grad()
        if prec:
            if self.jacobi:
                self.recompute_prec()
            else:
                super().recompute_prec()
        return
        
    def value(self):
        F  = self.a * super().value()
        F += self.b * self.goals.value()
        return F
    
    def _deriv_wrt_reconstructed_tensor(self) -> tensor:
        Yg = self.goals.gradient()
        Y = self.Mf - self.X
        Zb = (2*self.a)*Y + self.b*Yg
        return Zb

    def _tangent_reconstructed_tensor(self, V) -> tensor:
        Zd = super()._tangent_reconstructed_tensor(V, rescale=False)
        Md = tensor(Zd.data)
        Md = self.scaler.unscale_tensor(Md, shift=False).data
        Yd = self.goals.hessvec(Md)
        Zbd = (2*self.a)*Zd + self.b*Yd
        return Zbd

    def recompute_grad(self) -> None:
        # Parent class implementation relies on self._deriv_wrt_params,
        # which we've reimplemented.
        super().recompute_grad()
        return

    def recompute_prec(self) -> None:
        M = self.M
        tic = _time.time()
        goal_panels = self.goals.gn_diag_block_goal_updates()
        goal_panels[0] *= np.sqrt(self.b)
        toc = _time.time()
        self.times['gn_diag_block_goal_updates'].append(toc - tic)
        n = self._ndims
        grams = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(n)]
        self.C = []
        for k in range(n):
            D = np.array([1])
            for i in reversed(range(n)):
                if i != k:
                    D = np.kron(D, grams[i])
            G = M.core.to_tenmat(np.array([k])).data
            D = (2*self.a) * G @ D @ G.T
            block = linops.InvUpdatedKronPosDef([D, np.eye(self._shape[k])], goal_panels[k])
            self.C.append(block)
        grams[0] *= (2*self.a)
        kron_args = grams[::-1]
        B = linops.InvUpdatedKronPosDef(kron_args, goal_panels[-1])
        self.C.append(B)
        tic = _time.time()
        self.times['recompute_bj_prec, marginal'].append(tic - toc)
        self.block_jacobi_ops_cache.append([op for op in self.C])
        return
    
    def hessvec(self, V: ttensor) -> ttensor:
        # Parent class implementation relies on self._tangent_reconstructed_tensor,
        # which we've reimplemented.
        return super().hessvec(V)

    def compute_diag_blocks(self, M):
        # A helper function, for testing purposes only.
        # Computes the diagonal blocks of the Gauss-Newton Hessian with respect to the factor matrices
        # 
        S = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(self._ndims)]
        C = []
        goal_updates = self.goals.gn_diag_block_goal_updates()

        for n in range(self._ndims):
            D = np.array([1])
            for i in reversed(range(self._ndims)):
                if i != n:
                    D = np.kron(D, S[i])
            G = M.core.to_tenmat(np.array([n])).data
            D = (2*self.a) * G @ D @ G.T
            D = np.kron(D, np.eye(self._shape[n]))
            D += goal_updates[n] @ goal_updates[n].T
            C.append(D)

        D = np.array([1])
        for i in reversed(range(self._ndims)):
            D = np.kron(D, S[i])
        D = (2*self.a) * D
        D += goal_updates[-1] @ goal_updates[-1].T
        C.append(D)

        return C
