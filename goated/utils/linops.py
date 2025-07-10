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

    def item(self) -> np.floating:
        """
        If self.size == 1, we return a scalar representation of this linear operator.
        Otherwise, we raise an error.

        For type annotation purposes we say that .item() returns anything matching
        numpy.floating, but it will actually return a scalar of type self.dtype.
        """
        raise NotImplementedError()

    def __matmul__(self, other : np.ndarray) -> np.ndarray:
        """ Return self @ other """
        raise NotImplementedError()
    
    def __rmatmul__(self, other: np.ndarray) -> np.ndarray:
        """ Return other @ self """
        raise NotImplementedError()

    def to_array(self) -> np.ndarray:
        """Return an explicit representation of this operator as a real ndarray."""
        raise NotImplementedError()


def is_2d_square(arg) -> bool:
    if not hasattr(arg, 'shape'):
        return False
    if len(arg.shape) != 2:
        return False
    return arg.shape[0] == arg.shape[1]


RealLinOp_like = Union[RealLinOp, np.ndarray]


class InvTriangular(RealLinOp):
    """
    Represents inv(A), where A is a real triangular matrix given by a numpy ndarray.

    The action of this linear operator is implemented with the solve_triangular
    function in SciPy.linalg.
    """

    def __init__(self, A : np.ndarray, lower: bool, adjoint=None):
        assert is_2d_square(A)
        assert np.isrealobj(A)
        self.lower = lower
        self.A = A
        self._size  = A.size
        self._shape = A.shape
        self._dtype = A.dtype
        self._adjoint = InvTriangular(A.T, not self.lower, self) if adjoint is None else adjoint

    def item(self) -> np.floating:
        return 1 / self.A.item() # type: ignore

    def __matmul__(self,  other : np.ndarray) -> np.ndarray:
        return la.solve_triangular(self.A, other, trans=0, lower=self.lower, check_finite=False)
    
    def __rmatmul__(self, other : np.ndarray) -> np.ndarray:
        return la.solve_triangular(self.A, other.T, trans=1, lower=self.lower, check_finite=False).T

    def to_array(self) -> np.ndarray:
        return la.inv(self.A)


class InvPosDef(RealLinOp):
    """
    Represents inv(P), where P is a real positive definite matrix given by a numpy ndarray.

    A Cholesky decomposition of P is computed at construction time. The action of inv(P) is
    computed by the cho_solve function in SciPy.linalg.
    """

    def __init__(self, P: np.ndarray):
        assert is_2d_square(P)
        assert np.isrealobj(P)
        self.P = P
        self._size  = P.size
        self._shape = P.shape
        self._dtype = P.dtype
        self._chol = la.cho_factor(self.P)

    @property
    def T(self) -> Self:
        # override the default implementation, since we're self-adjoint.
        return self

    def item(self) -> np.floating:
        return 1 / self.P.item() # type: ignore
    
    def __matmul__(self,  other : np.ndarray) -> np.ndarray:
        return la.cho_solve(self._chol, other, check_finite=False)
    
    def __rmatmul__(self, other : np.ndarray) -> np.ndarray:
        temp = self.__matmul__(other.T)
        out = temp.T
        return out

    def to_array(self) -> np.ndarray:
        return la.inv(self.P)


class KronStructured(RealLinOp):
    """
    Represents the Kronecker product C = A ⨂ B of two real linear operators, (A, B),
    which can be either RealLinOps or (real) numpy ndarrays. A Kronecker product of
    n > 2 linear operators can be constructed with `KronStructured.recursive_dyadic`.

    This operator's action on vectors is implemented using a standard trick that
    avoids forming C explicitly. That action is extended to matrices with the help
    of the LinearOperator class from SciPy.sparse.
    """

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
        self._adjoint = KronStructured(A.T, B.T, adjoint=self) if adjoint is None else adjoint

    def item(self) -> np.floating:
        return self.A.item() * self.B.item() # type: ignore
    
    def _matvec(self, other: np.ndarray) -> np.ndarray:
        assert other.size == self.shape[1]
        if self._A_is_trivial:
            return self.A.item() * (self.B @ other) # type: ignore
        if self._B_is_trivial:
            return self.B.item() * (self.A @ other) # type: ignore
        inshape = other.shape
        out = self.B @ np.reshape(other, self._fwd_matvec_core_shape, order='F') @ self.A.T
        outshape = (-1, 1) if len(inshape) == 2 else (-1,)
        out = np.reshape(out, outshape, order='F')
        return out

    def _rmatvec(self, other: np.ndarray) -> np.ndarray:
        assert other.size == self.shape[0]
        if self._A_is_trivial:
            return self.A.item() * (self.B.T @ other) # type: ignore
        if self._B_is_trivial:
            return self.B.item() * (self.A.T @ other) # type: ignore
        inshape = other.shape
        out = self.B.T @ np.reshape(other, self._adj_matvec_core_shape, order='F') @ self.A
        outshape = (-1, 1) if len(inshape) == 2 else (-1,)
        out = np.reshape(out, outshape, order='F')
        return out

    def __matmul__(self, other : np.ndarray) -> np.ndarray:
        """ Return self @ other """
        return self._linop @ other # type: ignore
    
    def __rmatmul__(self, other: np.ndarray) -> np.ndarray:
        """ Return other @ self """
        return other @ self._linop  # type: ignore

    def to_array(self) -> np.ndarray:
        A_arr = self.A if isinstance(self.A, np.ndarray) else self.A.to_array()
        B_arr = self.B if isinstance(self.B, np.ndarray) else self.B.to_array()
        C = np.kron(A_arr, B_arr)
        return C

    @staticmethod
    def recursive_dyadic(kron_operands: Sequence[RealLinOp_like]) -> "KronStructured":
        assert len(kron_operands) > 1
        if len(kron_operands) == 2:
            out = KronStructured(kron_operands[0], kron_operands[1])
            return out
        arg = KronStructured.recursive_dyadic(kron_operands[1:])
        out = KronStructured(kron_operands[0], arg)
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

    def __init__(self, kron_factors : List[np.ndarray], U: np.ndarray, enable_to_array=True):
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
        self.invL = KronStructured.recursive_dyadic(invL_kron_factors)
        
        dim_update = U.shape[1]
        self.V = self.invL @ U
        self.W = np.eye(dim_update) + self.V.T @ self.V
        self.chol_W = la.cho_factor(self.W)
        if enable_to_array:
            self._kron_factors = kron_factors
            self._U = U
        else:
            self._kron_factors = None
            self._U = None
        self._size  = dim * dim
        self._shape = (dim, dim)
        self._dtype = self.invL.dtype
        return

    def to_array(self) -> np.ndarray:
        if self._U is None or self._kron_factors is None:
            msg = """
            The to_array method is not enabled for this InvUpdatedKronPosDef object.

            This method can only be enabled at construction time, by passing
            enable_to_array=True to the constructor.
            """
            raise ValueError(msg)
        assert self._U is not None
        assert self._kron_factors is not None
        K = np.eye(1)
        for kf in self._kron_factors:
            K = np.kron(K, kf)
        P = K + self._U @ self._U.T
        return la.inv(P)

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
