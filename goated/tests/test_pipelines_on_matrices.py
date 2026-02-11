from typing import Sequence
import numpy as np
import scipy.linalg as la
import unittest

from pyttb import tensor, ttensor, ktensor, cp_als  # type: ignore

from goated.cp import CPGoals, GocchaObjective
from goated.tucker import TuckerGoals, GotchaObjective
from goated.recipes.goals import FrobeniusGoal as SimpleGoal
import goated.utils.scalers as sc
import goated.rol_interface.models as rolm


def rel_fro_dist(A_approx, A):
    if not isinstance(A_approx, np.ndarray):
        A_approx = A_approx.double()
    if not isinstance(A, np.ndarray):
        A = A.double()
    abs_norm = la.norm((A_approx - A).ravel())
    rel      = la.norm(A.ravel())
    rel_norm = abs_norm / rel
    return rel_norm


class TestGocchaResolvesAmbiguity(unittest.TestCase):

    @staticmethod
    def degenerate_ktensor(dims, rank=2, seed=0) -> ktensor:
        gen = np.random.default_rng(seed)
        factor_matrices = []
        for d in dims:
            V = gen.standard_normal((d, rank))
            V = la.qr(V, mode='economic')[0]
            factor_matrices.append(V)
        out = ktensor(factor_matrices)
        return out

    def test_3x10_rank1_fullmatrix(self):
        dims = (3, 10)
        X_cp  = self.degenerate_ktensor(dims, rank=2)
        X     = tensor(X_cp.double())
        X_cp0 = ktensor([F[:,0].reshape((-1, 1)) for F in X_cp.factor_matrices ])
        X_cp1 = ktensor([F[:,1].reshape((-1, 1)) for F in X_cp.factor_matrices ])
        scaler = sc.Scaler(X, X.ndims - 2)
        Xs     = scaler.scale_tensor(X)

        np.random.seed(1997)
        U0s_cp, _, _ = cp_als(Xs, rank=1, maxiters=50, printitn=0)
        U0 = scaler.unscale_tensor(U0s_cp.full())

        if rel_fro_dist(X_cp0, U0) < rel_fro_dist(X_cp1, U0):
            X_goal = X_cp1
        else:
            X_goal = X_cp0

        before = rel_fro_dist(U0, X_goal)
        self.assertGreater(before, 0.1)
        # ^ Ensure our test case is meaningful.

        G0         = SimpleGoal(X_goal)
        goal_terms = CPGoals(scaler, [G0])
        goccha     = GocchaObjective(Xs, goal_terms, 1, 1)
        problem    = rolm.GoatedRolModel(goccha, U0s_cp)
        rol_params = rolm.build_cp_parameter_list()
        rol_params['General']['Output Level'] = 0
        rol_params['Status Test']['Iteration Limit'] = 500
        problem.solve(rol_params=rol_params)

        U1 = scaler.unscale_tensor(problem.decomp.full())
        after = rel_fro_dist(U1, X_goal)
        self.assertLessEqual(after, 1e-4)
        return
    

class TestGotchaResolvesAmbiguity(unittest.TestCase):

    @staticmethod
    def degenerate_ttensor(dims: Sequence[int], rank: Sequence[int], seed=0):
        gen = np.random.default_rng(seed)
        factor_matrices = []
        assert len(dims) == len(rank)
        assert all([r <= d for (r, d) in zip(rank, dims)])
        core = np.zeros(rank)
        for i, d in enumerate(dims):
            V = gen.standard_normal((d, rank[i]))
            V = la.qr(V, mode='economic')[0]
            factor_matrices.append(V)
            core[ core.ndim * (i,) ] = 1.0
        core = tensor(core)
        out = ttensor(core, factor_matrices)
        return out

    def test_3x10_rank1_fullmatrix(self):
        dims = (3, 10)
        X_tucker  = self.degenerate_ttensor(dims, rank=(2,2))
        X         = tensor(X_tucker.double())
        one2d     = tensor(np.atleast_2d(1.0))
        X_t0 = ttensor(one2d, [F[:,0].reshape((-1, 1)) for F in X_tucker.factor_matrices ])
        X_t1 = ttensor(one2d, [F[:,1].reshape((-1, 1)) for F in X_tucker.factor_matrices ])
        scaler = sc.Scaler(X, X.ndims - 2)
        Xs     = scaler.scale_tensor(X)

        U0 = X_t0
        U0s_tucker = scaler.scale_tensor(U0)
        X_goal = X_t1

        before = rel_fro_dist(U0, X_goal)
        self.assertGreater(before, 0.1)

        G0         = SimpleGoal(X_goal)
        goal_terms = TuckerGoals(scaler, [G0])
        goccha     = GotchaObjective(Xs, goal_terms, 1, 1)
        problem    = rolm.GoatedRolModel(goccha, U0s_tucker)
        rol_params = rolm.build_parameter_list()
        rol_params['General']['Output Level'] = 0
        rol_params['Status Test']['Iteration Limit'] = 50
        problem.solve(rol_params=rol_params)

        U1 = scaler.unscale_tensor(problem.decomp.full())
        after = rel_fro_dist(U1, X_goal)
        self.assertLessEqual(after, 1e-4)
        return



if __name__ == '__main__':
    unittest.main()
    print()
