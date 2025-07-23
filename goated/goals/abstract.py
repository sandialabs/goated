import numpy as np
from typing import Tuple, Optional


class Goal:

    def __init__(self, X, var, time, exo):
        self.var = var
        self.time = time
        self.exo = exo
        self.target = self.computeTarget(X)
        self.domain_shape = X.shape
        self.val : float = -1
        self.jac : np.ndarray = np.empty(())
        # ^ The last two instance variables are for caching.

    # Abstract function to be implemented in derived classes
    def computeTarget(self, X, compute_deriv=False) -> Tuple[float, np.ndarray] | float:
        raise NotImplementedError()
        
    def computeValue(self, U):
        val = self.computeTarget(U, compute_deriv=False)
        diff = val - self.target  # type: ignore
        F = np.sum(diff*diff)
        return F
    
    def computeDeriv(self, U):
        val, jac = self.computeTarget(U, compute_deriv=True)  # type: ignore
        diff = val - self.target  # type: ignore
        jac[np.ix_(range(jac.shape[0]),range(jac.shape[1]),self.var,self.time)] *= 2*np.reshape(diff,(1,1,1,len(self.time)))
        return jac
