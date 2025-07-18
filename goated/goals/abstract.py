import numpy as np


class Goal:

    def __init__(self, X, var, time, exo):
        self.var = var
        self.time = time
        self.exo = exo
        self.target = self.computeTarget(X)
        self.domain_shape = X.shape

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
