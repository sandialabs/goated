# try:
#     import pygenten as gt # Have to load GenTen before numpy to get OpenMP speedup
#     have_genten = True
# except:
#     have_genten = False

## NOTE: If I import pygenten as installed from pip then I get an unrecoverable error (a sigtrap).
#        So I'm just setting have_genten=False for now.
have_genten = False

import pyttb as ttb
import numpy as np


class Goal:

    def __init__(self, X, var, time, exo):
        self.var = var
        self.time = time
        self.exo = exo
        self.target = self.computeTarget(X)

    # Abstract function to be implemented in derived classes
    def computeTarget(self,X):
        return None
        
    def computeValue(self, U):
        val = self.computeTarget(U)
        diff = val - self.target
        F = np.sum(diff*diff)
        return F
    
    def computeDeriv(self, U):
        val,jac = self.computeTarget(U, compute_deriv=True)
        diff = val - self.target
        jac[np.ix_(range(jac.shape[0]),range(jac.shape[1]),self.var,self.time)] *= 2*np.reshape(diff,(1,1,1,len(self.time)))
        return jac
    
"""
Used in the run_gocp_opt notebook.
Not used in the run_gocp_rol notebook (or my copy, run_gocp_rol_riley).

"""
class CPGoal:
    def __init__(self, scaler, goals, weights):
        self.scaler = scaler
        self.goals = goals
        self.weights = weights
        
    def update(self, M):
        if have_genten and isinstance(M, gt.Ktensor):
            M = gt.make_ttb_ktensor(M,copy=False)
        self.Ms = self.scaler.unscale_ktensor(M)
        self.Mf = self.scaler.unscale_tensor(M.full())
        
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
        if have_genten and isinstance(M, gt.Ktensor):
            V = gt.make_gt_ktensor(V,copy=False)
        return V
