from pyttb import tensor, ttensor, ktensor  # type: ignore
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

    def update(self, M):
        self.Mf = M.full()
        
    def value(self):
        Y = self.Mf-self.X
        F = (Y.norm()**2)/self.s
        return F
    
    def gradient(self, M):
        Y = (2/self.s)*(self.Mf-self.X)
        G = Y.mttkrps(M)
        # ^ Why not use self.Mf, or store self.M?
        self.recompute_hess = True
        self.recompute_prec = True
        return G
    
    def hessvec(self, M, V):
        d = self._ndims
        A = M.factor_matrices
        r = A[0].shape[1]

        if self.recompute_hess:
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

        self.recompute_hess = False
        return Hv
    
    def precvec(self, M, V):
        d = self._ndims

        if self.recompute_prec:
            A = M.factor_matrices
            r = A[0].shape[1]
            #  Compute gram matrices
            S = [A[k].T@A[k] for k in range(d)]
            # diagonal factors
            self.U = np.ones((d,r,r))
            for k in range(d):
                for h in range(d):
                    if h != k:
                        self.U[k,:,:] *= S[h]
            # cholesky factors
            self.Vc = [la.cholesky(self.U[k,:,:], lower=False) for k in range(d)]

        #  Gauss-Newton block diagonal preconditioner
        Ab = V.factor_matrices
        Pv = []
        for k in range(d):
            tmp = la.solve_triangular(self.Vc[k], Ab[k].T, trans='T')
            tmp = la.solve_triangular(self.Vc[k], tmp, trans='N', overwrite_b=True)
            Pv.append((self.s/2)*tmp.T)
        Pv = ktensor(Pv)
        self.recompute_prec = False
        return Pv
    

class CPGoals:

    def __init__(self, scaler, goals : List[Goal], weights : Optional[Sequence[float]] = None):
        assert len(goals) > 0
        self.scaler = scaler
        self.goals = goals
        if weights is None:
            weights = np.ones((len(goals),), dtype=float)  # type: ignore
        self.weights = weights
        self.Mf : tensor  = None  # type: ignore
        self.Ms : ktensor = None  # type: ignore
        self._shape : Tuple[int,...] = goals[0].domain_shape
        self._ndim  : int = len(self._shape)
        self.recompute_hess = True
        
    def update(self, M : ktensor):
        self.Mf = self.scaler.unscale_tensor(M.full())
        self.Ms = self.scaler.unscale_ktensor(M)
        assert self.Mf.shape == self._shape
        assert self.Ms.shape == self._shape
        self.recompute_hess = True
        
    def value(self):
        F = 0
        for w,g in zip(self.weights, self.goals):
            F += w * g.computeValue(self.Mf)
        return F
    
    def gradient(self):
        Y = np.zeros(self._shape)
        for w,g in zip(self.weights, self.goals):
            Y += w * g.computeDeriv(self.Mf)
        Y = tensor(Y)
        V = Y.mttkrps(self.Ms)
        V = ktensor(V)
        V = self.scaler.unscale_ktensor(V)
        self.recompute_hess = True
        return V
    
    def hessvec(self, V):
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
            if self.recompute_hess:
                val, jac = g.computeTarget(self.Mf, compute_deriv=True) # type: ignore
                g.val = val
                g.jac = jac
            val = g.val
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

        self.recompute_hess = False
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
        
    def update(self, M):
        super().update(M)
        self.goals.update(M)
        
    def value(self):
        F  = self.a * super().value()
        F += self.b * self.goals.value()
        return F
    
    def gradient(self, M):
        G = super().gradient(M)
        # ^ That sets self.recompute_hess = self.recompute_prec = True.
        Ggoal = self.goals.gradient()
        G = [self.a*G[i]+self.b*Ggoal.factor_matrices[i] for i in range(self._ndims)]
        G = ktensor(G)
        return G
    
    def hessvec(self, M, V):
        Hv = super().hessvec(M,V)
        HvGoal = self.goals.hessvec(V)
        Hv = [self.a*Hv[i]+self.b*HvGoal.factor_matrices[i] for i in range(self._ndims)]
        Hv = ktensor(Hv)
        return Hv
    
    def precvec(self, M, V):
        Pv = super().precvec(M,V)
        return Pv
