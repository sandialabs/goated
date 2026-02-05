
from pyttb import tensor, ttensor, ktensor  # type: ignore
import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg as la

from goated.goals.abstract import TimeSeparableGoal



class FrobeniusGoal(TimeSeparableGoal):
    """
    A TimeSeparableGoal representation of a squared Frobenius norm penalty.
    This is mostly here for testing and demonstration purposes.

    Details
    -------
    A SomewhatSillyGoal object G stores a tensor T = G.ground_truth and a
    positive scalar s = G.shift. G's contribution to a tensor minimization
    objective is the penalty

        F(U) = \\sum_i { 
            || sqrt(s) - sqrt( s + || U[..., G.var, i] - T[..., G.var, i] ||_Fro^2 ) ||_2^2
        }.

    From a nonlinear least squares perspective, G is a vector-valued map
    with component functions

        G_i(U) = sqrt( s + || U[..., G.var, i] - T[..., G.var, i] ||_2^2 ).

    We need s > 0 so these component functions are differentiable at U=T.
    """

    def __init__(self,
            ground_truth: tensor | ttensor | ktensor,
            var:  ArrayLike | None = None,
            time: ArrayLike | None = None,
            shift=1e-14
        ):
        assert shift > 0
        self.shift = shift
        if var  is None:
            var  = np.arange(ground_truth.shape[-2])
        if time is None:
            time = np.arange(ground_truth.shape[-1]) 
        self.ground_truth = ground_truth.double()
        self._last_dim    = ground_truth.shape[-1]
        super().__init__(ground_truth, var, time)
        # ^ Sets self.target to a vector whose every component is sqrt(self.shift).
        return
    
    def computeVector(self, U: tensor | ttensor | ktensor, compute_deriv=False) -> tuple[np.ndarray, np.ndarray]:
        diff  = U.double() - self.ground_truth
        diff = diff[..., *self._var_cross_time]
        vec   = np.array([la.norm(diff[..., i])**2 for i in range(self.time.size)])
        norms = np.sqrt(self.shift + vec)
        if compute_deriv is True:
            jac = np.zeros(self.domain_shape)
            broadcast_helper = (self.domain_ndim - 2) * (None,)
            jac[..., *self._var_cross_time] = diff[..., :] / norms[*broadcast_helper, :]
        else:
            jac = np.empty(1)
        return (norms, jac)
