import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sparla
from typing import List, Union, Self, Literal, Tuple, Sequence


class RealLinOp:
    """
    This is an abstract class.
    
    Subclasses of RealLinOp represent linear operators in a way that's 
    polymorphic (for certain purposes) with real numpy ndarrays of ndim==2
    """

    __array_priority__ = 11
    # ^ The __array_priority__ static class variable determines how infix
    #   operators (most notably, @) are dispatched.
    #
    #   If Python initially dispatches the ndarray implementation of an infix
    #   operator, then the ndarray implementation will first check if the other
    #   operand has __array_priority__ greater than zero. If it does, then a
    #   function call like ndarray.__matmul__(array, other) will return the
    #   result of other.__rmatmul__(array).
    #   
    #   For our purposes, it should suffice to set __array_priority__ to 1.
    #   We set it to 11 in case someone does something weird and passes in 
    #   a numpy matrix (which behaves like an ndarray in many respects, and
    #   has __array_priority__ of 10). 
    #

    def __init__(self, *args, **kwargs) -> None:
        raise RuntimeError('RealLinOp.__init__ should never be called.')

    @property
    def ndim(self) -> Literal[2]:
        return 2

    @property
    def size(self) -> int:
        # This should equal int(prod(self.shape)), but our default implementation
        # stores this explicitly in a private variable called _size.
        return self._size # type: ignore

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape # type: ignore

    @property
    def dtype(self) -> np.dtype:
        return self._dtype # type: ignore

    @property
    def T(self) -> Self:
        """
        Return a RealLinOp that represents the adjoint of the current
        linear operator.
        
        If the current linear operator is self-adjoint, then we can
        literally just return `self`. Otherwise, we have to return
        another RealLinOp object that's suitably related to this object.
        Our default implementation assumes the RealLinOp representation
        of the adjoint is stored persistently in a private variable
        called _adjoint.
        """
        return self._adjoint # type: ignore

    def item(self):
        """
        If self.size == 1, we return a scalar representation of this linear operator.
        Otherwise, we raise an error.

        No default implementation is provided.
        """
        raise NotImplementedError()

    # The default implementations of __matmul__ and __rmatmul__ rely on
    # a private _linop member that overloads the matmul infix operator.
    #
    # It can be useful to have such a member if we only want to implement the
    # action of the linear operator (and its adjoint) on 1d ndarrays, as opposed
    # to implementing for ndarrays that could be 1d or 2d. In those situations
    # we can have _linop be a SciPy LinearOperator whose matvec and rmatvec
    # functions are instance methods defined in the ReaLinOp subclass at hand.

    def __matmul__(self, other : np.ndarray) -> np.ndarray:
        return self._linop @ other # type: ignore
    
    def __rmatmul__(self, other: np.ndarray) -> np.ndarray:
        return other @ self._linop # type: ignore


def is_2d_square(arg) -> bool:
    if not hasattr(arg, 'shape'):
        return False
    if len(arg.shape) != 2:
        return False
    return arg.shape[0] == arg.shape[1]


RealLinOp_like = Union[RealLinOp, np.ndarray]


class InvTriangular(RealLinOp):

    def __init__(self, A : np.ndarray, lower: bool, adjoint=None):
        assert is_2d_square(A)
        assert np.isrealobj(A)
        self.lower = lower
        self.A = A
        self._size  = A.size
        self._shape = A.shape
        self._dtype = A.dtype
        self._adjoint = InvTriangular(A.T, not self.lower, self) if adjoint is None else adjoint

    def item(self):
        return 1 / self.A.item()

    def __matmul__(self,  other : np.ndarray) -> np.ndarray:
        return la.solve_triangular(self.A, other, trans=0, lower=self.lower, check_finite=False)
    
    def __rmatmul__(self, other : np.ndarray) -> np.ndarray:
        return la.solve_triangular(self.A, other.T, trans=1, lower=self.lower, check_finite=False).T


class InvPosDef(RealLinOp):

    def __init__(self, A: np.ndarray):
        assert is_2d_square(A)
        assert np.isrealobj(A)
        self.A = A
        self._size  = A.size
        self._shape = A.shape
        self._dtype = A.dtype
        self._chol = la.cho_factor(self.A)

    @property
    def T(self) -> Self:
        # override the default implementation, since we're self-adjoint.
        return self

    def item(self):
        return 1 / self.A.item()
    
    def __matmul__(self,  other : np.ndarray) -> np.ndarray:
        return la.cho_solve(self._chol, other, check_finite=False)
    
    def __rmatmul__(self, other : np.ndarray) -> np.ndarray:
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

    def verify(self) -> None:
        """
        If P = LL' + U U', then this operator is supposed to represent M = inv(P).
        This function checks if self @ P is nearly the identity matrix.
        """
        explicit_K = np.eye(1)
        for kf in self.kron_factors:
            explicit_K = np.kron(explicit_K, kf)
        assert isinstance(self.U, np.ndarray)
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
            assert np.isrealobj(kf)
            K_cho_factors.append(la.cho_factor(kf, lower=True)[0])
            dim *= kf.shape[0]
        assert np.isrealobj(U)
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
            self.kron_factors = []
        self.verified = verify
        pass

    @property
    def T(self) -> Self:
        return self
    
    def __matmul__(self,  other : np.ndarray) -> np.ndarray:
        temp1 = self.invL @ other
        temp2 = self.V.T @ temp1
        temp3 = la.cho_solve(self.chol_W, temp2)
        temp4 = self.V @ temp3
        out = self.invL.T @ (temp1 - temp4)
        return out
    
    def __rmatmul__(self, other : np.ndarray) -> np.ndarray:
        # use the fact that we're self-adjoint.
        temp = self @ other.T
        out = temp.T
        return out


class DyadicKronStructured(RealLinOp):

    def __init__(self, A : RealLinOp_like, B : RealLinOp_like, adjoint=None):
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
        self._linop =  sparla.LinearOperator(
            dtype=self.dtype, shape=self.shape,
            matvec=self._matvec,   # type: ignore
            rmatvec=self._rmatvec  # type: ignore
        )
        self._adjoint = DyadicKronStructured(A.T, B.T, adjoint=self) if adjoint is None else adjoint

    def item(self):
        # This will raise a ValueError if self.size > 1.
        return self.A.item() * self.B.item()
    
    def _matvec(self, other: np.ndarray) -> np.ndarray:
        assert other.size == self.shape[1]
        if self._A_is_trivial:
            return self.A.item() * (self.B @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A @ other)
        inshape = other.shape
        out = self.B @ np.reshape(other, self._fwd_matvec_core_shape, order='F') @ self.A.T
        out = np.reshape(out, inshape, order='F')
        return out

    def _rmatvec(self, other: np.ndarray) -> np.ndarray:
        assert other.size == self.shape[0]
        if self._A_is_trivial:
            return self.A.item() * (self.B.T @ other)
        if self._B_is_trivial:
            return self.B.item() * (self.A.T @ other)
        inshape = other.shape
        out = self.B.T @ np.reshape(other, self._adj_matvec_core_shape, order='F') @ self.A
        out = np.reshape(out, inshape, order='F')
        return out
    
    @staticmethod
    def recursive_dyadic(kron_operands: Sequence[RealLinOp_like]) -> "DyadicKronStructured":
        assert len(kron_operands) > 1
        if len(kron_operands) == 2:
            out = DyadicKronStructured(kron_operands[0], kron_operands[1])
            return out
        arg = DyadicKronStructured.recursive_dyadic(kron_operands[1:])
        out = DyadicKronStructured(kron_operands[0], arg)
        return out


class KronStructured(RealLinOp):

    def __init__(self, kron_operands : Sequence[RealLinOp_like]):
        assert len(kron_operands) > 1
        self.kron_operands = kron_operands
        self.shapes = np.array([op.shape for op in kron_operands])
        self._shape = tuple(int(i) for i in np.prod(self.shapes, axis=0))
        forward = DyadicKronStructured.recursive_dyadic(self.kron_operands)
        self._linop   = forward._linop
        self._adjoint = forward.T
        self._dtype   = forward.dtype
