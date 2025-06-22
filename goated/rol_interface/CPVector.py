import copy
import numpy as np

from pyrol.pyrol import ROL
from pyrol.getTypeName import *

"""
This is not a vector of a single factor.
It's a vector of factors.
"""
class CPVector(getTypeName('Vector')):

    def __init__(self, factors):
        # factors are a list of NumPy arrays
        assert isinstance(factors,list)
        d = len(factors)
        for k in range(d):
            assert isinstance(factors[k],np.ndarray)
        super().__init__()
        self.data = factors
        self._dimension = self._get_dimension()

    def __getitem__(self, i):
        total = 0
        for ell in self.data:
            size = ell.size
            if i < total + size:
                flat = ell.reshape(-1)
                return flat[i - total]
            total += size

    def __setitem__(self, i, v):
        total = 0
        for ell in self.data:
            size = ell.size
            if i < total + size:
                flat = ell.reshape(-1)
                flat[i - total] = v
                return
            total += size

    def axpy(self, a, x):
        assert len(x.data) == len(self.data)
        d = len(self.data)
        for k in range(d):
            self.data[k] += a*x.data[k]

    def dimension(self):
        return self._dimension

    def dot(self, x):
        assert len(x.data) == len(self.data)
        d = len(self.data)
        ans = 0
        for k in range(d):
            ans += np.vdot(self.data[k],x.data[k])
        return ans

    def plus(self, x):
        assert len(x.data) == len(self.data)
        d = len(self.data)
        for k in range(d):
            self.data[k] += x.data[k]

    def scale(self, a):
        for ell in self.data:
            ell[:] *= a

    def setScalar(self, a):
        for ell in self.data:
            ell[:] = a

    # derived methods

    def applyBinary(self, op, x):
        assert x.dimension() == self.dimension()
        d = self.dimension()
        for k in range(d):
            self[k] = op.apply(self[k],x[k])

    def applyUnary(self, op):
        d = self.dimension()
        for k in range(d):
            self[k] = op.apply(self[k])

    def basis(self, i):
        b = self.clone()
        b.zero()
        b[i] = 1
        return b

    def clone(self):
        factors = copy.deepcopy(self.data)
        c = CPVector(factors)
        c.zero()  # workaround -- clone allocates but does not initialize
        return c

    def norm(self):
        return self.dot(self)**0.5

    def reduce(self, op):
        reduction_type = op.reductionType()
        match reduction_type:
            case ROL.Elementwise.REDUCE_MIN:
                ans = float('+inf')
                for ell in self.data:
                    ans = min(ans, ell.min())
            case ROL.Elementwise.REDUCE_MAX:
                ans = float('-inf')
                for ell in self.data:
                    ans = max(ans, ell.max())
            case ROL.Elementwise.REDUCE_SUM:
                ans = 0
                for ell in self.data:
                    ans += ell.sum()
            case ROL.Elementwise.REDUCE_AND:
                ans = True
                for ell in self.data:
                    ans = ans and ell.all()
                    if ans == False:
                        break
            case ROL.Elementwise.REDUCE_BOR:
                ans = 0
                d = self.dimension()
                for k in range(d):
                    ans = ans | int(self[k])
            case _:
                raise NotImplementedError(reduction_type)
        return ans

    def zero(self):
        self.setScalar(0)

    # private methods

    def _get_dimension(self):
        d = len(self.data)
        ans = 0
        for k in range(d):
            ans += self.data[k].size
        return ans

