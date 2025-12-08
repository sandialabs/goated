import numpy as np
from typing import Tuple
from pyttb import tensor, ttensor, ktensor  # type: ignore
from typing import TypeAlias

Tensor : TypeAlias = tensor | ttensor | ktensor


class Goal:
    # Abstract class. All implementations currently go through PhysicsGoal, defined below.

    def __init__(self, ground_truth : Tensor) -> None:
        self.target, _ = self.computeVector(ground_truth, compute_deriv=False)
        self.domain_shape = ground_truth.shape
        self.cached_vec  : np.ndarray = np.empty(())
        self.cached_jac  : np.ndarray = np.empty(())
        return
    
    def computeVector(self, U : Tensor, compute_deriv=False) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def computeScalar(self, U : Tensor) -> np.floating:
        """
        Return the squared Euclidean norm between this goal's vector-valued function
        evaluated at U and this goal's target (the vector-valued function evaluated
        at the ground truth).
        """
        vec, _ = self.computeVector(U, compute_deriv=False)
        diff = vec - self.target
        F = np.linalg.norm(diff)**2
        return F

    def computeGrad(self, U : Tensor) -> np.ndarray:
        """
        Return the gradient of self.computeScalar(U). This could be implemented with
        finite differences.
        """
        raise NotImplementedError()


class PhysicsGoal(Goal):

    def __init__(self, ground_truth : Tensor, var, time):
        if not isinstance(var, np.ndarray):
            var = np.array(var)
        if not isinstance(time, np.ndarray):
            time = np.array(time)
        self.var  : np.ndarray = var
        self.time : np.ndarray = time
        super().__init__(ground_truth)
        jac_m, jac_n = self.domain_shape[:2]
        i1 = np.arange(jac_m)  # arange of domain_shape[0]
        i2 = np.arange(jac_n)  # arange of domain_shape[1]
        i3 = self.var          # subset of arange domain_shape[2]
        i4 = self.time         # subset of arange domain_shape[3]
        self._grad_indices : tuple[np.ndarray,...] = np.ix_(i1, i2, i3, i4)
    
    def computeGrad(self, U : Tensor):
        vec, jac = self.computeVector(U, compute_deriv=True)
        # ^ It seems weird that vec.size is a different shape than jac.shape[-1].
        #   ... what is the derivative with respect to?
        diff = vec - self.target
        jac[self._grad_indices] *= 2*np.reshape(diff, (1,1,1) + self.target.shape)
        # Question: can I do away with the self._grad_indices, and just use
        #   broadcast_dims = (1,)*(jac.ndim - self.target.ndim)
        #   jac[:] *= 2*np.reshape(diff, broadcast_dims + self.target.shape)
        # ?
        return jac
