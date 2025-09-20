from pyttb import tensor, ktensor  # type: ignore
import numpy as np
import scipy.linalg as la
from goated.goals.abstract import Goal
from typing import Optional, Tuple, List, Sequence


class CPObjective:

    def __init__(self, X, s=None):
        self.X = X
        self.s = s if s is not None else  self.X.norm()**2
        self.Mf : Optional[tensor] = None
        self._shape : Tuple[int,...] = X.shape
        self._ndims : int = len(self._shape)
        self._grad : ktensor = ktensor()

    def update(self, M, prec=True, grad=True, hess=True) -> None:
        self.M = M
        self.Mf = M.full()
        if hess:
            self.recompute_hess()
        if prec:
            self.recompute_prec()
        if grad:
            self.recompute_grad()
        return
    
    def recompute_hess(self) -> None:
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
        # off-diagonal factors
        self.Ub = np.ones((d,d,r,r))
        for k in range(d):
            for l in range(d):
                for h in range(d):
                    if h != l and h != k:
                        self.Ub[k,l,:,:] *= S[h]
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
    
    def hessvec(self, V) -> ktensor:
        d = self._ndims
        A = self.M.factor_matrices
        r = A[0].shape[1]
        Ab = V.factor_matrices
        Sb = [Ab[k].T @ A[k] for k in range(d)]
        Hv = []
        for k in range(d):
            # accumulate off-diagonal factors
            Ukb = np.zeros((r, r))
            for l in range(d):
                if l != k:
                    Ukb += self.Ub[k,l,:,:]*Sb[l]
            # Gauss-Newton Hessian-vector product
            Hv.append((2/self.s) * (Ab[k] @ self.U[k] + A[k] @ Ukb))
        Hv = ktensor(Hv)
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


class CPGoals:

    def __init__(self, scaler, goals : List[Goal], weights : Optional[Sequence[float] | np.ndarray] = None):
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
            val, jac = g.computeTarget(self.Mf, compute_deriv=True)
            g.val = val
            g.jac = jac
        return

    def value(self) -> float:
        F = 0
        for w,g in zip(self.weights, self.goals):
            F += w * g.computeValue(self.Mf)
        return F
    
    def recompute_grad(self) -> None:
        Y = np.zeros(self._shape)
        for w,g in zip(self.weights, self.goals):
            Y += w * g.computeDeriv(self.Mf)
        Y = tensor(Y)
        V = Y.mttkrps(self.Ms)
        V = ktensor(V)
        V = self.scaler.unscale_ktensor(V)
        self._grad = V

    def gradient(self) -> ktensor:
        return self._grad

    def hessvec(self, V: ktensor) -> ktensor:
        # Compute unscaled data if we were provided scaling
        Vs = self.scaler.unscale_ktensor(V)

        # form ktensors with M.u{k} replaced by V.u{k}
        d = self._ndim
        Mt = []
        for k in range(d):
            Mt_k = self.Ms.copy()
            Mt_k.factor_matrices[k] = Vs.factor_matrices[k].copy()
            Mt.append(Mt_k)

        # compute full M dot tensor
        Md = np.zeros(self._shape, order='F')
        for MM in Mt:
            Md += MM.full().double()

        # compute necessary gradient info
        Yd = np.zeros(self._shape,order='F')
        for w,g in zip(self.weights,self.goals):
            var = g.var
            time = g.time
            num_time = len(time)
            jac = g.jac
                
            # compute val_dot (could tangent-differentiate fcn, but since we already have jac, we just do a mat-vec)
            val_dot = np.zeros((num_time,1))
            for i in range(num_time):
                val_dot[i] = np.reshape(jac[:,:,var,time[i]],(1,-1),order='F') @ np.reshape(Md[:,:,var,time[i]],(-1,1),order='F')

            # compute dot gradient tensor dF/dM(i,j,v,t)
            # adding in 2*diff(i)*goal_scaling(i) * the tangent derivative of jac_M would
            # make this the full Hessian-vector product
            jac_dot = np.zeros(jac.shape)
            for i in range(num_time):
                jac_dot[:,:,var,time[i]] = (2*val_dot[i])*jac[:,:,var,time[i]]

            Yd += w*jac_dot

        # compute unscaled Gauss-Newton Hessian-vector product
        Yd = tensor(Yd)
        Hv = Yd.mttkrps(self.Ms)

        # transform back to scaled variables
        Hv = self.scaler.unscale_ktensor(ktensor(Hv))
        return Hv

    def eval_goals(self, U: tensor, scaled=False):   
        if scaled:
            U = self.scaler.unscale_tensor(U)
        v = np.array([g.computeValue(U) for g in self.goals])
        return v 

    def auto_reweight(self, U: tensor, scaled=False):
        v = self.eval_goals(U, scaled=scaled)
        self.weights = 1 / (v * len(self.goals))
        return


class GocchaObjective(CPObjective):

    def __init__(self, X, goal : CPGoals, a, b):
        super().__init__(X, s=1.0)
        self.goals = goal
        self.a = a
        self.b = b
        
    def update(self, M, grad=True, prec=True, hess=True):
        super().update(M, prec=prec, grad=False, hess=hess)
        self.goals.update(M)
        if grad:
            self.recompute_grad()
        
    def value(self) -> float:
        F  = self.a * super().value()
        F += self.b * self.goals.value()
        return F
    
    def recompute_grad(self):
        super().recompute_grad()
        G = self._grad.factor_matrices
        Ggoal = self.goals.gradient()
        G = [self.a*G[i]+self.b*Ggoal.factor_matrices[i] for i in range(self._ndims)]
        G = ktensor(G)
        self._grad = G
    
    def hessvec(self, V):
        HvFrob_factors = super().hessvec(V).factor_matrices
        HvGoal_factors = self.goals.hessvec(V).factor_matrices
        Hv = [self.a*F + self.b*G for (F,G) in zip(HvFrob_factors, HvGoal_factors)]
        Hv = ktensor(Hv)
        return Hv
    
    def precvec(self, V):
        Pv = super().precvec(V)
        return Pv
