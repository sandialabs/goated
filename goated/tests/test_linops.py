import unittest
import numpy as np
import scipy.linalg as la
from goated.utils import linops


def fro_norm_comparison_tol(array : np.ndarray):
    dtype, nrm = array.dtype, la.norm(array)
    rel_tol = np.finfo(dtype).eps * nrm
    abs_tol = np.finfo(dtype).eps ** 0.5
    tol = max(rel_tol, abs_tol)  # type: ignore
    return tol


class TestRealLinOps(unittest.TestCase):

    def check_adjoint_toarray(self, op: linops.RealLinOp):
        fwd_array = op.to_array()
        adj_array = op.T.to_array()
        tol = fro_norm_comparison_tol(fwd_array)
        self.assertLessEqual(la.norm(fwd_array - adj_array.T), tol)
        return

    def check_forward(self, op : linops.RealLinOp):
        m, n = op.shape
        op_array_actual = op.to_array()
        Im = np.eye(m)
        In = np.eye(n)
        op_array_expect_1 = op @ In
        op_array_expect_2 = Im @ op
        tol = fro_norm_comparison_tol(op_array_actual)
        self.assertLessEqual(la.norm(op_array_expect_1 - op_array_actual), tol)
        self.assertLessEqual(la.norm(op_array_expect_2 - op_array_actual), tol)
        return
    
    def check_adjoint(self, op: linops.RealLinOp):
        adj_array_actual = op.to_array().T
        # ^ Using op.to_array().T instead of op.T.to_array() makes
        #   check_adjoint(op) ever so slightly different than check_forward(op.T)
        n, m = adj_array_actual.shape
        Im = np.eye(m)
        In = np.eye(n)
        adj_array_expect_1 = op.T @ Im
        adj_array_expect_2 = In @ op.T
        tol = fro_norm_comparison_tol(adj_array_actual)
        self.assertLessEqual(la.norm(adj_array_expect_1 - adj_array_actual), tol)
        self.assertLessEqual(la.norm(adj_array_expect_2 - adj_array_actual), tol)
        return

    def test_inv_triangular(self):
        g = np.random.default_rng(0)
        X = g.standard_normal(size=(5, 6))
        X = X @ X.T
        X_lower = np.tril(X)
        L = linops.InvTriangular(X_lower, lower=True)
        self.check_forward(L)
        self.check_adjoint(L)
        self.check_adjoint_toarray(L)
        return

    def test_inv_posdef(self):
        g = np.random.default_rng(0)
        X = g.standard_normal(size=(5, 6))
        X = X @ X.T
        X_linop = linops.InvPosDef(X)
        self.check_forward(X_linop)
        self.check_adjoint_toarray(X_linop)
        return

    def test_kron_structured_from_arrays(self):
        g = np.random.default_rng(0)
        for m, n in [(1,1), (1,5), (5,1), (3,3), (4,2)]:

            X = g.standard_normal((m, n))

            C_XX  = linops.KronStructured(X, X)
            self.check_forward(C_XX)
            self.check_adjoint(C_XX)
            self.check_adjoint_toarray(C_XX)

            C_XXT = linops.KronStructured(X, X.T)
            self.check_forward(C_XXT)
            self.check_adjoint(C_XXT)
            self.check_adjoint_toarray(C_XXT)

            A = g.standard_normal(X.shape)
            B = g.standard_normal((7, 5))

            for Y in [A, B]:
                C_XY = linops.KronStructured(X, Y)
                self.check_forward(C_XY)
                self.check_adjoint(C_XY)
                self.check_adjoint_toarray(C_XY)

                C_YX = linops.KronStructured(Y, X)
                self.check_forward(C_YX)
                self.check_adjoint(C_YX)
                self.check_adjoint_toarray(C_YX)
        return
    
    def test_kron_structured_from_reallinops(self):
        g = np.random.default_rng(0)
        X = g.standard_normal(size=(5, 6))
        X = X @ X.T
        X_lower = np.tril(X)
        L = linops.InvTriangular(X_lower, lower=True)
        P = linops.InvPosDef(X)

        C_LP = linops.KronStructured(L, P)
        self.check_forward(C_LP)
        self.check_adjoint(C_LP)
        self.check_adjoint_toarray(C_LP)

        C_PL = linops.KronStructured(P, L)
        self.check_forward(C_PL)
        self.check_adjoint(C_PL)
        self.check_adjoint_toarray(C_PL)
        return

    def test_kron_structured_recursive(self):
        g = np.random.default_rng(0)
        X = g.standard_normal(size=(5, 6))
        X = X @ X.T
        X_lower = np.tril(X)
        L = linops.InvTriangular(X_lower, lower=True)
        P = linops.InvPosDef(X)
        C = linops.KronStructured.recursive_dyadic([L, L.T, P])
        self.check_forward(C)
        self.check_adjoint(C)
        self.check_adjoint_toarray(C)
        return
    
    def test_inv_updated_kron_pos_def(self):
        g = np.random.default_rng(0)
        kron_args = []
        dims = [4, 7, 5]
        for dim in dims:
            X = g.standard_normal(size=(dim, dim+1))
            X = X @ X.T 
            kron_args.append(X)
        n = np.prod(dims)
        
        update_sizes = [2, n+1]
        for s in update_sizes:

            U = g.uniform(size=(n, s))
            M = linops.InvUpdatedKronPosDef(kron_args, U)
            self.check_forward(M)
            self.check_adjoint(M)
            self.check_adjoint_toarray(M)

            U[:,0] = 0
            M = linops.InvUpdatedKronPosDef(kron_args, U)
            self.check_forward(M)
            self.check_adjoint(M)
            self.check_adjoint_toarray(M)  
        return


if __name__ == '__main__':
    runner = TestRealLinOps()

    runner.test_kron_structured_from_arrays()
    runner.test_kron_structured_from_reallinops()
    runner.test_kron_structured_recursive()

    runner.test_inv_posdef()
    runner.test_inv_triangular()
    runner.test_inv_updated_kron_pos_def()

    print()  # keep me for breakpoint convience reasons.
