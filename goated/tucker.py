from pyttb import tensor, ttensor  # type: ignore
import numpy as np

import goated.utils.linops as linops
from goated.abstractobj import LowRankObjective
from collections import defaultdict
from goated.goals.abstract import Goal, PhysicsGoal, TuckerGoals
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

    def _forward(self) -> None:
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

    def _collect_backproped(self, blocks) -> ttensor:
        return ttensor(blocks[-1], blocks[:-1])

    def _tangent_reconstructed_tensor(self, V: ttensor, rescale=True) -> tensor:
        M = self.M
        Zd = M.core.ttm(V.factor_matrices[0],0) + V.core.ttm(M.factor_matrices[0],0)
        for i in range(1, self._ndims):
            Zd = self.Z[i-1].ttm(V.factor_matrices[i],i) + Zd.ttm(M.factor_matrices[i], i)
        if rescale:
            Zd *= 2/self.s
        return Zd  # type: ignore

    def recompute_prec(self) -> None:
        tic = _time.time()
        n = self._ndims
        M = self.M
        grams = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(n)]
        self.C = []
        for k in range(n):
            # construct a linear operator that can act as the inverse of the diagonal
            # block in our Gauss-Newton Hessian that corresponds to the k-th non-core
            # ttensor factor.
            D = np.array([1])
            for i in reversed(range(n)):
                if i != k:
                    D = np.kron(D, grams[i])
            G = M.core.to_tenmat(np.array([k])).data
            DG = (2/self.s) * G @ D @ G.T
            block = linops.InvPosDef(DG)
            self.C.append(block)
        # construct an implicit inverse of the diagonal block in the Gauss-Newton Hessian
        # that corresponds to the ttensor's core factor.
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
            Pvk = Vk @ Ck
            # ^ That's equivalent to vec(Pvk) := (Ck \tensor Ik) vec(Vk), where I_k
            #   is the identity matrix of size equal to the number of columns in Vk.
            Pv.append(Pvk)

        Pvc = V.core.data.ravel(order='F')
        Pvc = self.C[-1] @ Pvc
        Pvc = Pvc.reshape(V.core.shape, order='F')
        Pv  = ttensor(tensor(Pvc), Pv)
        toc = _time.time()
        self.times['gn_bd_precvec'].append(toc - tic)
        return Pv


class GotchaObjective(TuckerObjective):

    def __init__(self, X,  goals : TuckerGoals, a, b, jacobi=True):
        super().__init__(X, s=1.0)
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
            self.recompute_prec()
        return
        
    def value(self) -> float:
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
            g._tucker_gn_block_diag_goal_updates(w, self.M, self.goals.scaler, factor_cols)
        factors = [np.column_stack(cols) for cols in factor_cols]
        return factors

    def recompute_prec(self) -> None:
        if not self.jacobi:
            super().recompute_prec()
            return
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
            # ^ That operator acts via left-multiplication on VECTORIZED non-core ttensor factors;
            #   this is in contrast to the operators in TuckerObjective.recompute_prec, which end
            #   up acting via right-multiplication on non-core ttensor factors without vectorization.
            #   
            #   The behavior of TuckerObjective's recompute_prec and precvec is equivalent to what
            #   we do here (and in GotchaObjective.precvec) when goal_panels[k] is zero.
            #
            self.C.append(block)
        grams[0] *= (2*self.a)
        kron_args = grams[::-1]
        B = linops.InvUpdatedKronPosDef(kron_args, goal_panels[-1])
        self.C.append(B)
        tic = _time.time()
        self.times['recompute_bj_prec, marginal'].append(tic - toc)
        self.block_jacobi_ops_cache.append([op for op in self.C])
        return
    
    def precvec(self, V: ttensor) -> ttensor:        
        tic = _time.time()
        if not self.jacobi:
            return super().precvec(V)
        Pv = []
        for k in range(self._ndims):
            Ck = self.C[k]
            Vk = V.factor_matrices[k]
            Pvk = Ck @ Vk.ravel(order='F')
            Pvk = Pvk.reshape(Vk.shape, order='F')
            Pv.append(Pvk)

        # The following lines are duplicated with TuckerObjective.precvec.
        Pvc = V.core.data.ravel(order='F')
        Pvc = self.C[-1] @ Pvc
        Pvc = Pvc.reshape(V.core.shape, order='F')
        Pv  = ttensor(tensor(Pvc), Pv)
        toc = _time.time()
        self.times['gn_bd_precvec'].append(toc - tic)
        return Pv
 
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
        Yd = self.goals.hessvec_wrt_reconstruction(Md)
        Zbd = (2*self.a)*Zd + self.b*Yd
        return Zbd

    # Methods for debugging

    def compute_diag_blocks(self, M=None) -> List[np.ndarray]:
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
