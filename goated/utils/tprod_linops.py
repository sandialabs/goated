import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sparla
from typing import List


class RealLinOp:
    
    # Function implementations below are merely defaults.
    # Don't hesitate to override them if need be.

    __array_priority__ = 100

    @property
    def ndim(self):
        return 2

    @property
    def size(self):
        return self._size

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def T(self):
        return self._adjoint

    def item(self):
        # If self.size == 1, return a scalar representation of this linear operator.
        # Otherwise, error.
        raise NotImplementedError()

    def __matmul__(self, other):
        return self._linop @ other
    
    def __rmatmul__(self, other):
        return other @ self._linop


def is_2d_square(arg):
    if not hasattr(arg, 'shape'):
        return False
    if len(arg.shape) != 2:
        return False
    return arg.shape[0] == arg.shape[1]


class InvTriangular(RealLinOp):
    """
    NOTE: can avoid relying on sparla.LinearOperator since we can implement matmul and rmatmul directly.
    """

    def __init__(self, A : np.ndarray, lower: bool, adjoint=None):
        assert is_2d_square(A)
        self.lower = lower
        self.A = A
        self._size  = A.shape[0]**2
        self._shape = A.shape
        self._dtype = A.dtype
        self._adjoint = InvTriangular(A.T, not self.lower, self) if adjoint is None else adjoint

    def item(self):
        return 1 / self.A.item()

    def __matmul__(self, other):
        return la.solve_triangular(self.A, other, trans=0, lower=self.lower, check_finite=False)
    
    def __rmatmul__(self, other):
        return la.solve_triangular(self.A, other.T, trans=1, lower=self.lower, check_finite=False).T


class InvPosDef(RealLinOp):
    """
    NOTE: can avoid relying on sparla.LinearOperator since we can implement matmul and rmatmul directly.
    """

    def __init__(self, A: np.ndarray):
        assert is_2d_square(A)
        self.A = A
        self._size  = A.shape[0]**2
        self._shape = A.shape
        self._dtype = A.dtype
        self._chol = la.cho_factor(self.A)

    @property
    def T(self):
        # override the default implementation, since we're self-adjoint.
        return self

    def item(self):
        return 1 / self.A.item()
    
    def __matmul__(self, other):
        return la.cho_solve(self._chol, other, check_finite=False)
    
    def __rmatmul__(self, other):
        temp = self.__matmul__(other.T)
        out = temp.T
        return out
    

class InvUpdatedKronPosDef(RealLinOp):
    """
    A representation of a positive definite linear operator
    
        M = inv( K + U U' ),
    
    where K is a positive definite matrix with known Kronecker product
    structure and U is a tall-and-thin matrix.

    This linear operator's action is implemented by precomputing some
    intermediate quantities at construction time and then using those
    quantifies in the Woodbury matrix identity. Specifically, we precompute

        1. an implicit representation of L = cho_factor(K, lower=True),
        2. an explicit representation of V = inv(L) @ U,
        3. a factored  representation of W = I + V'V,
    
    and then we use the formula 

        M = inv(L') (I - V @ inv(W) @ V') @ inv(L).
    
    The essence of this method can be preserved with different factorizations
    for K. For example, instead of computing L = cho_factor(K, lower=True),
    we could compute P = pinv(sqrtm(K)) and substitute P wherever inv(L) or
    inv(L') were used.
    """

    def verify(self):
        """
        If P = LL' + U U', then this operator is supposed to represent M = inv(P).
        This function checks if self @ P is nearly the identity matrix.
        """
        explicit_K = np.eye(1)
        for kf in self.kron_factors:
            explicit_K = np.kron(explicit_K, kf)
        explicit_P = explicit_K + self.U @ self.U.T
        expect_I = self @ explicit_P
        nrmP = la.norm(explicit_P)
        I = np.eye(self.shape[0])
        rel_tol = np.finfo(self.dtype).eps * nrmP
        abs_tol = np.finfo(self.dtype).eps ** 0.5
        tol = max(rel_tol, abs_tol)
        assert la.norm(I - expect_I) <= tol
        

    def __init__(self, kron_factors : List[np.ndarray], U: np.ndarray, verify=False):
        K_cho_factors = []
        dim = 1
        for kf in kron_factors:
            K_cho_factors.append(la.cho_factor(kf, lower=True)[0])
            dim *= kf.shape[0]
        assert dim == U.shape[0]
        self.K_cho_factors = K_cho_factors
        invL_kron_factors = [InvTriangular(lf, lower=True) for lf in K_cho_factors]
        self.invL = KronStructured(invL_kron_factors)
        
        dim_update = U.shape[1]
        self.V = self.invL @ U
        self.W = np.eye(dim_update) + self.V.T @ self.V
        self.chol_W = la.cho_factor(self.W)
        self._size  = dim * dim
        self._shape = (dim, dim)
        self._dtype = self.invL.dtype
        if verify:
            self.kron_factors = kron_factors
            self.U = U
            self.verify()
        else:
            self.U = None
            self.kron_factors = None
        self.verified = verify
        pass

    @property
    def T(self):
        return self
    
    def __matmul__(self, other):
        temp1 = self.invL @ other
        temp2 = self.V.T @ temp1
        temp3 = la.cho_solve(self.chol_W, temp2)
        temp4 = self.V @ temp3
        out = self.invL.T @ (temp1 - temp4)
        return out
    
    def __rmatmul__(self, other):
        # use the fact that we're self-adjoint.
        temp = self @ other.T
        out = temp.T
        return out


class DyadicKronStructed(RealLinOp):

    def __init__(self, A, B, adjoint=None):
        assert A.ndim == 2
        assert B.ndim == 2
        self.A = A
        self.B = B
        self._A_is_trivial = A.size == 1
        self._B_is_trivial = B.size == 1
        self._shape = ( A.shape[0]*B.shape[0], A.shape[1]*B.shape[1] )
        self._size = self.shape[0] * self.shape[1]
        self._fwd_matvec_core_shape = (B.shape[1], A.shape[1])
        self._adj_matvec_core_shape = (B.shape[0], A.shape[0])
        self._dtype = A.dtype
        self._linop =  sparla.LinearOperator(dtype=self.dtype, shape=self.shape, matvec=self.matvec, rmatvec=self.rmatvec)
        self._adjoint = DyadicKronStructed(A.T, B.T, adjoint=self) if adjoint is None else adjoint

    def item(self):
        # This will raise a ValueError if self.size > 1.
        return self.A.item() * self.B.item()
    
    def matvec(self, other):
        inshape = other.shape
        assert other.size == self.shape[1]
        if self._A_is_trivial:
            return self.A.item() * (self.B @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A @ other)
        out = self.B @ np.reshape(other, self._fwd_matvec_core_shape, order='F') @ self.A.T
        out = np.reshape(out, inshape, order='F')
        return out

    def rmatvec(self, other):
        inshape = other.shape
        assert other.size == self.shape[0]
        if self._A_is_trivial:
            return self.A.item() * (self.B.T @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A.T @ other)
        out = self.B.T @ np.reshape(other, self._adj_matvec_core_shape, order='F') @ self.A
        out = np.reshape(out, inshape, order='F')
        return out
    
    @staticmethod
    def build_polyadic(kron_operands):
        if len(kron_operands) == 2:
            out = DyadicKronStructed(kron_operands[0], kron_operands[1])
            return out
        # else, recurse
        arg = DyadicKronStructed.build_polyadic(kron_operands[1:])
        out = DyadicKronStructed(kron_operands[0], arg)
        return out


class KronStructured(RealLinOp):

    def __init__(self, kron_operands):
        self.kron_operands = kron_operands
        assert all([op.ndim == 2 for op in kron_operands])
        self.shapes = np.array([op.shape for op in kron_operands])
        self._shape = tuple(int(i) for i in np.prod(self.shapes, axis=0))
        forward = DyadicKronStructed.build_polyadic(self.kron_operands)
        self._linop   = forward._linop
        self._adjoint = forward.T
        self._dtype = self.kron_operands[0].dtype
