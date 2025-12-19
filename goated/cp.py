from pyttb import tensor, ktensor  # type: ignore
import numpy as np
import scipy.linalg as la
from goated.goals.abstract import Goal, PhysicsGoal, CPGoals
from typing import Optional, Tuple, List, Sequence, Optional
from goated.abstractobj import LowRankObjective


class CPObjective(LowRankObjective):

    def __init__(self, X, s=None):
        self.X = X
        self.s = s if s is not None else  self.X.norm()**2
        self.M  : ktensor = ktensor()
        self.Mf : tensor  = tensor()
        self._shape : Tuple[int,...] = X.shape
        self._ndims : int = len(self._shape)
        self._grad : ktensor = ktensor()
    
    def _forward(self) -> None:
        self.Mf = self.M.full()
        return
    
    def _backprop(self, Zb: tensor):
        # one call to the built‐in MTTKRP for each mode
        return Zb.mttkrps(self.M)

    def _collect_backproped(self, blocks):
        return ktensor(blocks)
    
    def _tangent_reconstructed_tensor(self, V: ktensor, rescale=True) -> tensor:
        """
        Compute the forward-mode directional derivative of the full CP  
        reconstruction M.full() in the direction V (another ktensor).
        
        If rescale=True it also multiplies by (2/self.s) so that you get
        exactly the d/dh of the Frobenius-part of the Gauss-Newton model.
        """
        M = self.M
        Zd = tensor(np.zeros(self._shape))
        for k in range(self._ndims):
            tmp_factors = [Ai for Ai in M.factor_matrices]
            tmp_factors[k] = V.factor_matrices[k]
            tmp = ktensor(tmp_factors)
            Zd += tmp.full()
        
        if rescale:
            Zd *= (2.0/self.s)
        
        return Zd  # type: ignore
    
    def recompute_prec(self):
        d = self._ndims
        A = self.M.factor_matrices
        r = A[0].shape[1]
        #  Compute gram matrices
        S = [A[k].T @ A[k] for k in range(d)]
        # diagonal factors
        U = np.ones((d,r,r))
        for k in range(d):
            for h in range(d):
                if h != k:
                    U[k,:,:] *= S[h]
        # cholesky factors
        self.Vc = [la.cholesky(U[k,:,:], lower=False) for k in range(d)]
        return

    def precvec(self, V: ktensor) -> ktensor:
        d = self._ndims
        #  Gauss-Newton block diagonal preconditioner
        Ab = V.factor_matrices
        Pv = []
        for k in range(d):
            tmp = la.solve_triangular(self.Vc[k], Ab[k].T, trans='T')
            tmp = la.solve_triangular(self.Vc[k], tmp, trans='N', overwrite_b=True)
            Pv.append((self.s/2)*tmp.T)
        Pv = ktensor(Pv)
        return Pv


class GocchaObjective(CPObjective):

    def __init__(self, X, goals : CPGoals, a, b):
        super().__init__(X, s=1.0)
        self.goals  = goals
        self.scaler = goals.scaler
        self.a = a
        self.b = b
        
    def update(self, M: ktensor, grad=True, prec=True):
        super().update(M, prec=prec, grad=False)
        self.goals.update(M, self.Mf, grad=True, jacs=True)
        if grad:
            self.recompute_grad()
        
    def value(self) -> float:
        F  = self.a * super().value()
        F += self.b * self.goals.value()
        return F
    
    def recompute_grad(self) -> None:
        # Parent class implementation relies on self._deriv_wrt_params,
        # which we've reimplemented.
        return super().recompute_grad()

    def hessvec(self, V: ktensor) -> ktensor:
        return super().hessvec(V)
    
    def precvec(self, V: ktensor) -> ktensor:
        Pv = super().precvec(V)
        return Pv

    def _deriv_wrt_reconstructed_tensor(self) -> tensor:
        Yg = self.goals.gradient_wrt_reconstruction()
        Y = self.Mf - self.X
        Zb = (2*self.a)*Y + self.b*Yg
        return Zb
    
    def _tangent_reconstructed_tensor(self, V) -> tensor:
        Zd = super()._tangent_reconstructed_tensor(V, rescale=False)
        Md = tensor(Zd.data)
        Md = self.scaler.unscale_tensor(Md, shift=False).data
        Yd = self.goals.hessvec_wrt_reconstruction(Md)
        Zbd = (2*self.a)*Zd + self.b*Yd
        return Zbd
