import pyttb as ttb
import numpy as np

# import goated.utils.exo as ex
# import goated.utils.scalers as sc


class CPObjective:

    def __init__(self, X, s=None):
        self.X = X
        self.s = s
        if self.s is None:
            self.s = self.X.norm()**2
        
    def update(self, M):
        self.Mf = M.full()
        
    def value(self, M):
        Y = self.Mf-self.X
        F = (Y.norm()**2)/self.s
        return F
    
    def gradient(self, M):
        Y = (2/self.s)*(self.Mf-self.X)
        G = Y.mttkrps(M)
        self.recompute_hess = True
        self.recompute_prec = True
        return G
    
    def hessvec(self, M, V):
        d = M.ndims
        A = M.factor_matrices
        Ab = V.factor_matrices
    
        #  Compute gram matrices
        if self.recompute_hess:
            self.S = [A[k].T@A[k] for k in range(d)]
        Sb = [Ab[k].T@A[k] for k in range(d)]
        R = A[0].shape[1]

        if self.recompute_hess:
            # diagonal factors
            self.U = np.ones((d,R,R))
            for k in range(d):
                for h in range(d):
                    if h != k:
                        self.U[k,:,:] *= self.S[h]

            # off-diagonal factors
            self.Ub = np.ones((d,d,R,R))
            for k in range(d):
                for l in range(d):
                    for h in range(d):
                        if h != l and h != k:
                            self.Ub[k,l,:,:] *= self.S[h]

        Hv = [None]*d
        for k in range(d):
            # accumulate off-diagonal factors
            Ukb = np.zeros((R,R))
            for l in range(d):
                if l != k:
                    Ukb += self.Ub[k,l,:,:]*Sb[l]

            # Gauss-Newton Hessian-vector product
            Hv[k] = (2/self.s)*(Ab[k]@self.U[k] + A[k]@Ukb)

        self.recompute_hess = False
        return Hv
    
    def precvec(self, M, V):
        import scipy.linalg as lin

        d = M.ndims
        A = M.factor_matrices
        Ab = V.factor_matrices
    
        #  Compute gram matrices
        if self.recompute_prec:
            self.S = [A[k].T@A[k] for k in range(d)]
        Sb = [Ab[k].T@A[k] for k in range(d)]
        R = A[0].shape[1]

        # diagonal factors
        if self.recompute_prec:
            self.U = np.ones((d,R,R))
            for k in range(d):
                for h in range(d):
                    if h != k:
                        self.U[k,:,:] *= self.S[h]

            # cholesky factors
            self.Vc = [np.linalg.cholesky(self.U[k,:,:],upper=True) for k in range(d)]

        #  Gauss-Newton block diagonal preconditioner
        Pv = [None]*d
        for k in range(d):
            tmp = lin.solve_triangular(self.Vc[k], Ab[k].T, trans='T')
            tmp = lin.solve_triangular(self.Vc[k], tmp, trans='N', overwrite_b=True)
            Pv[k] = (self.s/2)*tmp.T

        self.recompute_prec = False
        return Pv
    
"""
A helper class that gets passed to GocchaObjective.
Why is the name what it is? We'll never know ...

See also: goated.goals.abstract::CPGoal.

It seems that CPGoal2 serves the exact same purpose
as CPGoal, with the exception that CPGoal is only used
in run_gocp_opt.ipynb while run_gocp_rol.ipynb uses CPGoal2.

So I can probably use CPGoal2 as a replacement for
goated.goals.abstract::CPGoal and then remove it from
this file.
"""
class CPGoal2:

    def __init__(self, scaler, goals, weights):
        self.scaler = scaler
        self.goals = goals
        self.weights = weights
        
    def update(self, M):
        self.Mf = self.scaler.unscale_tensor(M.full())
        self.Ms = self.scaler.unscale_ktensor(M)
        
    def value(self, M):
        F = 0
        for w,g in zip(self.weights,self.goals):
            F += w * g.computeValue(self.Mf)
        return F
    
    def gradient(self, M):
        Y = np.zeros(M.shape)
        for w,g in zip(self.weights,self.goals):
            Y += w * g.computeDeriv(self.Mf)
        Y = ttb.tensor(Y)
        V = Y.mttkrps(self.Ms)
        V = ttb.ktensor(V)
        V = self.scaler.unscale_ktensor(V)
        self.recompute_hess = True
        return V
    
    def hessvec(self, M, V):
        # Compute unscaled data if we were provided scaling
        Vs = self.scaler.unscale_ktensor(V)

        # form ktensors with M.u{k} replaced by V.u{k}
        d = M.ndims
        Mt = [None]*d
        for k in range(d):
            Mt[k] = self.Ms.copy()
            Mt[k].factor_matrices[k] = Vs.factor_matrices[k].copy()

        # compute full M dot tensor
        Md = np.zeros(self.Ms.shape,order='F')
        for MM in Mt:
            Md += MM.full().double()

        # compute necessary gradient info
        Yd = np.zeros(self.Ms.shape,order='F')
        for w,g in zip(self.weights,self.goals):
            var = g.var
            time = g.time
            num_time = len(time)
            if self.recompute_hess:
                val,jac = g.computeTarget(self.Mf, compute_deriv=True)
                setattr(g,'val',val)
                setattr(g,'jac',jac)
            else:
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
        Yd = ttb.tensor(Yd)
        Hv = Yd.mttkrps(self.Ms)

        # transform back to scaled variables
        Hv = self.scaler.unscale_ktensor(ttb.ktensor(Hv))

        self.recompute_hess = False
        return Hv


class GocchaObjective(CPObjective):

    def __init__(self, X, goal, a, b):
        super().__init__(X, s=1.0)
        self.goal = goal
        self.a = a
        self.b = b
        
    def update(self, M):
        super().update(M)
        self.goal.update(M)
        
    def value(self, M):
        F = self.a*super().value(M)
        F += self.b*self.goal.value(M)
        return F
    
    def gradient(self, M):
        G = super().gradient(M)
        Ggoal = self.goal.gradient(M)
        G = [self.a*G[i]+self.b*Ggoal.factor_matrices[i] for i in range(M.ndims)]
        return G
    
    def hessvec(self, M, V):
        Hv = super().hessvec(M,V)
        HvGoal = self.goal.hessvec(M,V)
        Hv = [self.a*Hv[i]+self.b*HvGoal.factor_matrices[i] for i in range(M.ndims)]
        return Hv
    
    def precvec(self, M, V):
        Pv = super().precvec(M,V)
        return Pv
