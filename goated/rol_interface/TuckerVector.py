import copy
import numpy as np

from pyrol.pyrol import ROL
from pyrol.getTypeName import *


class TuckerVector(getTypeName('Vector')):

    def __init__(self, core, factors):
        # core is a NumPy array and factors are a list of NumPy arrays
        assert isinstance(core,np.ndarray)
        assert isinstance(factors,list)
        d = len(factors)
        for k in range(d):
            assert isinstance(factors[k],np.ndarray)
        super().__init__()
        self.core = core
        self.factors = factors
        self._dimension = self._get_dimension()

    def __getitem__(self, i):
        size = self.core.size
        if i < size:
            ind = np.unravel_index(i,shape=self.core.shape,order='F')
            return self.core[ind]
        total = size
        for ell in self.factors:
            size = ell.size
            if i < total + size:
                ind = np.unravel_index(i-total,shape=ell.shape,order='F')
                return ell[ind]
            total += size

    def __setitem__(self, i, v):
        size = self.core.size
        if i < size:
            ind = np.unravel_index(i,shape=self.core.shape,order='F')
            self.core[ind] = v
            return
        total = size
        for ell in self.factors:
            size = ell.size
            if i < total + size:
                ind = np.unravel_index(i-total,shape=ell.shape,order='F')
                ell[ind] = v
                return
            total += size

    def axpy(self, a, x):
        assert len(x.factors) == len(self.factors)
        d = len(self.factors)
        self.core += a*x.core
        for k in range(d):
            self.factors[k] += a*x.factors[k]

    def dimension(self):
        return self._dimension

    def dot(self, x):
        assert len(x.factors) == len(self.factors)
        d = len(self.factors)
        ans = np.vdot(self.core,x.core)
        for k in range(d):
            ans += np.vdot(self.factors[k],x.factors[k])
        return ans

    def plus(self, x):
        assert len(x.factors) == len(self.factors)
        d = len(self.factors)
        self.core += x.core
        for k in range(d):
            self.factors[k] += x.factors[k]

    def scale(self, a):
        self.core[:] *= a
        for ell in self.factors:
            ell[:] *= a

    def setScalar(self, a):
        self.core[:] = a
        for ell in self.factors:
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
        core = copy.deepcopy(self.core)
        factors = copy.deepcopy(self.factors)
        c = TuckerVector(core, factors)
        c.zero()  # workaround -- clone allocates but does not initialize
        return c

    def norm(self):
        return self.dot(self)**0.5

    def reduce(self, op):
        reduction_type = op.reductionType()
        match reduction_type:
            case ROL.Elementwise.REDUCE_MIN:
                ans = self.core.min()
                for ell in self.factors:
                    ans = min(ans, ell.min())
            case ROL.Elementwise.REDUCE_MAX:
                ans = self.core.max()
                for ell in self.factors:
                    ans = max(ans, ell.max())
            case ROL.Elementwise.REDUCE_SUM:
                ans = self.core.sum()
                for ell in self.factors:
                    ans += ell.sum()
            case ROL.Elementwise.REDUCE_AND:
                ans = self.core.all()
                if ans == False:
                    return ans
                for ell in self.factors:
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
        d = len(self.factors)
        ans = self.core.size
        for k in range(d):
            ans += self.factors[k].size
        return ans

