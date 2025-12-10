from pyttb import tensor, ktensor  # type: ignore
import numpy as np
import scipy.linalg as la
from goated.goals.abstract import Goal, PhysicsGoal
from typing import Optional, Tuple, List, Sequence


class CPObjective:

    def __init__(self, X, s=None):
        self.X = X
        self.s = s if s is not None else  self.X.norm()**2
        self.Mf : Optional[tensor] = None
        self._shape : Tuple[int,...] = X.shape
        self._ndims : int = len(self._shape)
        self._grad : ktensor = ktensor()

    def update(self, M, prec=True, grad=True) -> None:
        self.M = M
        self.Mf = M.full()
        if prec:
            self.recompute_prec()
        if grad:
            self.recompute_grad()
        return
        
    def value(self) -> float:
        Y = self.Mf-self.X
        F = (Y.norm()**2)/self.s
        return F
    
    def recompute_grad(self) -> None:
        Y : tensor = (2/self.s)*(self.Mf-self.X)
        G = ktensor(Y.mttkrps(self.M))
        self._grad = G

    def gradient(self) -> ktensor:
        return self._grad
    
    def hessvec(self, V: ktensor) -> ktensor:
        Zbd = self._tangent_reconstructed_tensor(V)
        Hv_factors = [Zbd.mttkrp(self.M, k) for k in range(self._ndims)]
        Hv = ktensor(Hv_factors)
        return Hv
    
    def recompute_prec(self):
        d = self._ndims
        A = self.M.factor_matrices
        r = A[0].shape[1]
        #  Compute gram matrices
        S = [A[k].T @ A[k] for k in range(d)]
        # diagonal factors
        self.U = np.ones((d,r,r))
        for k in range(d):
            for h in range(d):
                if h != k:
                    self.U[k,:,:] *= S[h]
        # cholesky factors
        self.Vc = [la.cholesky(self.U[k,:,:], lower=False) for k in range(d)]
        return

    def precvec(self, V) -> ktensor:
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

    def _deriv_wrt_reconstructed_tensor(self) -> tensor:
        Zb = (2/self.s)*(self.Mf-self.X)
        return Zb
    
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


class CPGoals:

    def __init__(self, scaler, goals : List[PhysicsGoal], weights : Optional[Sequence[float] | np.ndarray] = None):
        assert len(goals) > 0
        self.scaler = scaler
        self.goals = goals
        if weights is None:
            weights = np.ones((len(goals),), dtype=float)
        elif not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        self.weights : np.ndarray = weights
        self.Mf : tensor  = tensor()
        self.Ms : ktensor = ktensor()
        self._shape : Tuple[int,...] = goals[0].domain_shape
        self._ndim  : int = len(self._shape)
        self._grad  : ktensor = ktensor()
        
    def update(self, M : ktensor, grad=True, jacs=True):
        self.Mf = self.scaler.unscale_tensor(M.full())
        self.Ms = self.scaler.unscale_ktensor(M)
        assert self.Mf.shape == self._shape
        assert self.Ms.shape == self._shape
        if grad:
            self.recompute_grad()
        if jacs:
            self.recompute_jacs()

    def recompute_jacs(self) -> None:
        for _,g in zip(self.weights,self.goals):
            val, jac = g.computeVector(self.Mf, compute_deriv=True)
            g.cached_vec = val
            g.cached_jac = jac
        return

    def value(self) -> float:
        F = 0
        for w,g in zip(self.weights, self.goals):
            F += w * g.computeScalar(self.Mf)
        return F
    
    def recompute_grad(self) -> None:
        Y = np.zeros(self._shape)
        for w,g in zip(self.weights, self.goals):
            Y += w * g.computeGrad(self.Mf)
        Y = tensor(Y)
        V = Y.mttkrps(self.Ms)
        V = ktensor(V)
        V = self.scaler.unscale_ktensor(V)
        self._grad = V

    def gradient(self) -> ktensor:
        return self._grad

    def _sort_of_hessvec(self, Md: tensor) -> tensor:
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


class GocchaObjective(CPObjective):

    def __init__(self, X, goals : CPGoals, a, b):
        super().__init__(X, s=1.0)
        self.goals  = goals
        self.scaler = goals.scaler
        self.a = a
        self.b = b
        
    def update(self, M: ktensor, grad=True, prec=True):
        super().update(M, prec=prec, grad=False)
        self.goals.update(M, grad=True, jacs=True)
        if grad:
            self.recompute_grad()
        
    def value(self) -> float:
        F  = self.a * super().value()
        F += self.b * self.goals.value()
        return F
    
    def recompute_grad(self) -> None:
        super().recompute_grad()
        G = self._grad.factor_matrices
        Ggoal = self.goals.gradient()
        G = [self.a*G[i]+self.b*Ggoal.factor_matrices[i] for i in range(self._ndims)]
        G = ktensor(G)
        self._grad = G
    
    def _tangent_reconstructed_tensor(self, V) -> tensor:
        Zd = super()._tangent_reconstructed_tensor(V, rescale=False)
        Md = tensor(Zd.data)
        Md = self.scaler.unscale_tensor(Md, shift=False).data
        Yd = self.goals._sort_of_hessvec(Md)
        Zbd = (2*self.a)*Zd + self.b*Yd
        return Zbd

    def hessvec(self, V: ktensor) -> ktensor:
        return super().hessvec(V)
    
    def precvec(self, V: ktensor) -> ktensor:
        Pv = super().precvec(V)
        return Pv
