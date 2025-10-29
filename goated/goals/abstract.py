import numpy as np
from typing import Tuple
from pyttb import tensor, ttensor, ktensor  # type: ignore
from typing import TypeAlias

Tensor : TypeAlias = tensor | ttensor | ktensor


class Goal:

    def __init__(self, ground_truth : Tensor) -> None:
        self.target, _ = self.computeTarget(ground_truth, compute_deriv=False)
        self.domain_shape = ground_truth.shape
        return
    
    def computeTarget(self, U : Tensor, compute_deriv=False) -> Tuple[float, np.ndarray]:
        # Abstract function to be implemented in derived classes
        raise NotImplementedError()
    
    def computeValue(self, U : Tensor):
        val, _ = self.computeTarget(U, compute_deriv=False)
        diff = val - self.target  # type: ignore
        F = np.linalg.norm(diff)**2
        return F

    def computeDeriv(self, U : Tensor):
        val, jac = self.computeTarget(U, compute_deriv=True)
        diff = val - self.target  # type: ignore
        # jac[self._grad_indices] *= 2*np.reshape(diff, (1,1,1) + self.target.shape)
        # return jac
        raise NotImplementedError()


class PhysicsGoal(Goal):

    def __init__(self, ground_truth : Tensor, var, time):
        if not isinstance(var, np.ndarray):
            var = np.array(var)
        if not isinstance(time, np.ndarray):
            time = np.array(time)
        self.var = var
        self.time = time
        self.val : float = -1
        self.jac : np.ndarray = np.empty(())
        # ^ The last two instance variables can be used by other classes for caching.
        super().__init__(ground_truth)
        jac_m, jac_n = self.domain_shape[:2]
        i1 = np.arange(jac_m)  # arange of domain_shape[0]
        i2 = np.arange(jac_n)  # arange of domain_shape[1]
        i3 = self.var          # subset of arange domain_shape[2]
        i4 = self.time         # subset of arange domain_shape[3]
        self._grad_indices : tuple[np.ndarray,...] = np.ix_(i1, i2, i3, i4)
        
    def computeValue(self, U : Tensor):
        val, _ = self.computeTarget(U, compute_deriv=False)
        diff = val - self.target  # type: ignore
        F = np.linalg.norm(diff)**2
        return F
    
    def computeDeriv(self, U : Tensor):
        val, jac = self.computeTarget(U, compute_deriv=True)
        # ^ It seems weird that val.size is a different shape than jac.shape[-1].
        #   ... what is the derivative with respect to?
        diff = val - self.target  # type: ignore
        jac[self._grad_indices] *= 2*np.reshape(diff, (1,1,1) + self.target.shape)
        # Question: can I do away with the self._grad_indices, and just use
        #   broadcast_dims = (1,)*(jac.ndim - self.target.ndim)
        #   jac[:] *= 2*np.reshape(diff, broadcast_dims + self.target.shape)
        # ?
        return jac
