from pyttb import tensor, ttensor  # type: ignore
import numpy as np

import goated.utils.linops as linops
from goated.abstractobj import LowRankObjective
from collections import defaultdict
from goated.goals.abstract import Goal, PhysicsGoal
from typing import Tuple, List, Optional, Sequence
import time as _time


class TuckerObjective(LowRankObjective):

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

    def _forward(self):
        ZZ : tensor = self.M.core.ttm(self.M.factor_matrices[0], 0) # type: ignore
        self.Z = [ ZZ ]
        for i in range(1,self._ndims):
            ZZ = ZZ.ttm(self.M.factor_matrices[i], i)
            self.Z.append(ZZ)
        self.Mf = ZZ # the same as M.full().
        self.Zt = [ self.M.core.to_tenmat(np.array([0])).data.T ]
        for i in range(1,self._ndims):
            self.Zt.append(self.Z[i-1].to_tenmat(np.array([i])).data.T)

    def _backprop(self, Zb: tensor) -> List:
        Gf : list[np.ndarray | tensor] = [None] * self._ndims # type: ignore
        # ^ We need to reserve space for Gf since we fill it in reverse order.
        for i in reversed(range(self._ndims)):
            Gf[i] = Zb.to_tenmat(np.array([i])).data @ self.Zt[i]
            Zb = Zb.ttm(self.M.factor_matrices[i].T,i)
        Gf.append(Zb)
        return Gf

    def _collect_backproped(self, blocks):
        return ttensor(blocks[-1], blocks[:-1])

    def _tangent_reconstructed_tensor(self, V: ttensor, rescale=True) -> tensor:
        M = self.M
        Zd = M.core.ttm(V.factor_matrices[0],0) + V.core.ttm(M.factor_matrices[0],0)
        for i in range(1, self._ndims):
            Zd = self.Z[i-1].ttm(V.factor_matrices[i],i) + Zd.ttm(M.factor_matrices[i], i)
        if rescale:
            Zd *= 2/self.s
        return Zd  # type: ignore

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

    def precvec(self, V: ttensor) -> ttensor:        
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
    """
    Constituent PhysicsGoal objects define their targets in terms 
    of the un-scaled tensor, but GotchaObjective works in terms of
    a scaled tensor. So this class maintains its own scaler.
    """

    def __init__(self, scaler, goals : List[PhysicsGoal], weights : Optional[Sequence[float] | np.ndarray] = None, _shape=None):
        self.scaler = scaler
        self.goals = goals
        if weights is None:
            weights = np.ones((len(self.goals),))
        self.weights = weights
        self.M   : ttensor = ttensor()
        self.Mfs : tensor  = tensor()
        self._shape : Tuple[int,...] = goals[0].domain_shape
        self._ndim  : int = len(self._shape)
        self._grad  : tensor = tensor()
        self.jac_times : list[float] = []
        return
        
    def update(self, M: ttensor, Mf: tensor, grad=True, jacs: bool=True, prec=True) -> None:
        self.M = M
        self.Mfs = self.scaler.unscale_tensor(Mf)
        if grad or jacs or prec:
            self.recompute_jacs()
        if grad:
            self.recompute_grad(use_cached_jacs=True)
        return
    
    def recompute_jacs(self) -> None:
        for _, g in zip(self.weights, self.goals):
            tic = _time.time()
            vec, jac = g.computeVector(self.Mfs, compute_deriv=True)
            g.cached_vec = vec
            g.cached_jac = jac
            toc = _time.time()
            self.jac_times.append(toc - tic)
        return

    def value(self) -> float:
        F = 0.0
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

    # same for CP and Tucker; trivial
    def gradient_wrt_reconstruction(self) -> tensor:
        return self._grad
    
    # same for CP and Tucker
    def _sort_of_hessvec(self, Md: tensor) -> tensor:
        Yd = np.zeros(self._shape, order='F')
        for w,g in zip(self.weights, self.goals):
            jac_dot = g._gn_hessvec(Md)
            Yd += w*jac_dot
        Yd = self.scaler.unscale_tensor(Yd, shift=False)
        return Yd

    def eval_goals(self, U: tensor, scaled=False) -> np.ndarray:   
        if scaled:
            U = self.scaler.unscale_tensor(U)
        v = np.array([g.computeScalar(U) for g in self.goals])
        return v 

    def auto_reweight(self, U: tensor, scaled=False) -> None:
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
        jac_times = []
        self.goals.update(
            self.M, self.Mf,
            grad=(prec or grad),
            jacs=(prec or grad)
        )
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
    
    def recompute_grad(self) -> None:
        # Parent class implementation relies on self._deriv_wrt_params,
        # which we've reimplemented.
        super().recompute_grad()
        return
    
    def gn_diag_block_goal_updates(self) -> list[np.ndarray]:
        factor_cols = [[] for _ in range(self.goals._ndim + 1)]
        for w, g in zip(self.goals.weights, self.goals.goals):
            g._tucker_gn_block_diag_goal_updates(w, self.M, self.scaler, factor_cols)
        factors = [np.column_stack(cols) for cols in factor_cols]
        return factors

    def recompute_prec(self) -> None:
        M = self.M
        tic = _time.time()
        goal_panels = self.gn_diag_block_goal_updates()
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

    def _deriv_wrt_reconstructed_tensor(self) -> tensor:
        Yg = self.goals.gradient_wrt_reconstruction()
        Y = self.Mf - self.X
        Zb = (2*self.a)*Y + self.b*Yg
        return Zb

    def _tangent_reconstructed_tensor(self, V: ttensor) -> tensor:
        Zd = super()._tangent_reconstructed_tensor(V, rescale=False)
        Md = tensor(Zd.data)
        Md = self.scaler.unscale_tensor(Md, shift=False).data
        Yd = self.goals._sort_of_hessvec(Md)
        Zbd = (2*self.a)*Zd + self.b*Yd
        return Zbd

    def compute_diag_blocks(self, M=None):
        # A helper function, for testing purposes only.
        # Computes the diagonal blocks of the Gauss-Newton Hessian with respect to the factor matrices
        # 
        M0 = self.M
        if M is None:
            M = M0
            restore = False
        else:
            restore = True
            self.update(M, True, True)

        S = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(self._ndims)]
        C = []
        goal_updates = self.gn_diag_block_goal_updates()

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
        
        if restore:
            self.update(M0, True, True)

        return C
