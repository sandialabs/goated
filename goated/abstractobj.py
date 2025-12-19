
from abc import ABC, abstractmethod
from pyttb import tensor
from typing import List, Tuple

class LowRankObjective(ABC):
    """
    Common Gauss-Newton driver for any low-rank format.
    Subclasses must implement:
      _forward()                fill in self.Mf (full tensor) and any extra caches,
      _backprop(Zb) → List[...]  take a full-tensor Zb and return list of factor-gradient blocks,
      recompute_prec()           build self.C so that precvec(V) uses it,
    and can override:
      _tangent_reconstructed_tensor(V)
      _deriv_wrt_reconstructed_tensor()
    """
    def __init__(self, X, s=None):
        self.X     = X
        self.s     = s if s is not None else X.norm()**2
        self.M     = None           # filled by update()
        self.Mf    = None           # full-tensor, filled by _forward
        self._grad = None
        self.C     = None           # block-diag precond (format depends on subclass)

    def update(self, M, *, grad=True, prec=True) -> None:
        self.M = M
        self._forward()
        if grad:
            self.recompute_grad()
        if prec:
            self.recompute_prec()
        return

    @abstractmethod
    def _forward(self):
        """ Given the low-rank self.M, fill in self.Mf and any pre-computed
            intermediates needed by _backprop(). """
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

    def gradient(self):
        return self._grad

    def hessvec(self, V):
        # 1) forward-mode: directional derivative of the recon
        Zbd = self._tangent_reconstructed_tensor(V)
        # 2) back-propagate into factor space
        blocks = self._backprop(Zbd)
        # 3) package
        return self._collect_backproped(blocks)

    def precvec(self, V):
        # just delegate to block-diag preconditioner built in recompute_prec()
        raise NotImplementedError()

    def _deriv_wrt_reconstructed_tensor(self):
        # default for least squares term:  (2/s)*(Mf - X)
        return (2.0/self.s)*(self.Mf - self.X)

    def _tangent_reconstructed_tensor(self, V):
        # default is: nothing!  subclasses can override to do
        # the “tangent of full-reconstruction” step
        raise NotImplementedError()

    @abstractmethod
    def _backprop(self, Zb) -> List:
        """
        Given an adjoint/full-tensor Zb, return a list of arrays
        corresponding to the derivative blocks for each factor (and
        core if needed).
        """
        pass

    @abstractmethod
    def _collect_backproped(self, blocks):
        """
        Given the list of factor-gradients (and maybe a core-grad),
        assemble them into the same ttensor/ktensor-type as self.M.
        """
        pass

    @abstractmethod
    def recompute_prec(self):
        """ Build self.C so that _apply_prec(V) can solve each block. """
        pass

