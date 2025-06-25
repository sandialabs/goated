import pyttb as ttb
import numpy as np
from copy import deepcopy


import logging
class ExtraCopyFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith("Selected no copy, but input data isn't")
logger = logging.getLogger()  # root logger
logger.addFilter(ExtraCopyFilter())


import pyrol
from pyrol.pyrol import ROL


class TuckerVector(pyrol.Vector):

    @staticmethod
    def from_ttensor(x, copy=False):
        if copy:
            return TuckerVector(deepcopy(x.core.data), deepcopy(x.factor_matrices))
        else:
            return TuckerVector(x.core.data, x.factor_matrices)
        
    def to_ttensor(self, copy=False):
        return ttb.ttensor(ttb.tensor(self.core), self.factors, copy=copy)
    
    def to_numpy_1d(self):
        y = np.reshape(self.core, (-1,), order='F')
        for f in self.factors:
            y = np.concatenate([y, np.reshape(f, (-1,), order='F')])
        return y

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
        core = deepcopy(self.core)
        factors = deepcopy(self.factors)
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


class CPVector(pyrol.Vector):

    @staticmethod
    def from_ktensor(x, copy=False):
        if copy:
            return CPVector(deepcopy(x.factor_matrices))
        else:
            return CPVector(x.factor_matrices)
        
    def to_ktensor(self, copy=False):
        return ttb.ktensor(self.data, copy=copy)

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
        factors = deepcopy(self.data)
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

