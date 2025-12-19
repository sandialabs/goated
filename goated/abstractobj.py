
from abc import ABC, abstractmethod
from pyttb import tensor, ttensor, ktensor
from typing import List, Tuple, TypeVar


LowRankTensor_t = TypeVar('LowRankTensor_t', ktensor, ttensor)


class LowRankObjective(ABC):
    """
    Common Gauss-Newton driver for any low-rank format.
    Subclasses must implement:
      _forward()                fill in self.Mf (full tensor) and any extra caches,
      _backprop(Zb) → List[...]  take a full-tensor Zb and return list of factor-gradient blocks,
      _tangent_reconstructed_tensor(V)
    and can override:
      _deriv_wrt_reconstructed_tensor()
      recompute_prec()
    """
    def __init__(self, X: tensor, s=None):
        self.X     = X
        self.s     = s if s is not None else X.norm()**2
        self.M     = None           # filled by update()
        self.Mf    = tensor()       # full-tensor, filled by _forward
        self._grad = None
        return

    def update(self, M: LowRankTensor_t, *, grad=True, prec=True) -> None:
        self.M = M
        self._forward()
        if grad:
            self.recompute_grad()
        if prec:
            self.recompute_prec()
        return

    @abstractmethod
    def _forward(self) -> None:
        """
        Given the low-rank self.M, fill in self.Mf and any pre-computed
        intermediates needed by _backprop().
        """
        pass

    @abstractmethod
    def _tangent_reconstructed_tensor(self, V: LowRankTensor_t) -> tensor:
        """ 
        TODO: write docstring
        """
        pass

    @abstractmethod
    def _backprop(self, Zb) -> List:
        """
        Given an adjoint/full-tensor Zb, return a list of arrays
        corresponding to the derivative blocks for each factor (and
        core if needed).
        """
        pass

    @abstractmethod
    def _collect_backproped(self, blocks) -> LowRankTensor_t:
        """
        Given the list of factor-gradients (and maybe a core-grad),
        assemble them into the same ttensor/ktensor-type as self.M.
        """
        pass

    def value(self) -> float:
        # F = (||Mf - X||^2) / s
        diff = self.Mf - self.X
        return diff.norm()**2 / self.s

    def recompute_grad(self):
        # 1) form the adjoint wrt the reconstructed tensor
        Zb = self._deriv_wrt_reconstructed_tensor()
        # 2) back-propagate into factor space
        blocks = self._backprop(Zb)
        # 3) package into the same low-rank format as self.M
        self._grad = self._collect_backproped(blocks)

    def gradient(self) -> LowRankTensor_t:
        return self._grad # type: ignore

    def hessvec(self, V: LowRankTensor_t) -> LowRankTensor_t:
        # 1) forward-mode: directional derivative of the recon
        Zbd = self._tangent_reconstructed_tensor(V)
        # 2) back-propagate into factor space
        blocks = self._backprop(Zbd)
        # 3) package
        return self._collect_backproped(blocks)

    def precvec(self, V: LowRankTensor_t) -> LowRankTensor_t:
        return V

    def _deriv_wrt_reconstructed_tensor(self) -> tensor:
        # default for least squares term:  (2/s)*(Mf - X)
        return (2.0/self.s)*(self.Mf - self.X)

    def recompute_prec(self) -> None:
        # The default preconditioner is trivial, so there's
        # no work to be done here.
        return

