import pyrol

import goated.rol_interface.objective as gro
import goated.rol_interface.vectorization as vu
from goated.tucker import GotchaObjective, TuckerObjective
from goated.cp import GocchaObjective, CPObjective
from typing import Union


def build_parameter_list(output_level=1, status_test_iter_limit=10, general_krylov_iter_limit=2000):
    params = pyrol.ParameterList()
    params['General'] =  pyrol.ParameterList()
    params['General']['Output Level'] = output_level
    params['General']['Krylov'] = pyrol.ParameterList()
    params['General']['Krylov']['Iteration Limit'] = general_krylov_iter_limit
    params['Step'] = pyrol.ParameterList()
    params['Step']['Trust Region'] = pyrol.ParameterList()
    params['Step']['Trust Region']['Initial Radius'] = 10.0
    params['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
    params['Status Test'] = pyrol.ParameterList()
    params['Status Test']['Iteration Limit'] = status_test_iter_limit
    return params


def build_cp_parameter_list():
    params = pyrol.ParameterList()
    params['General'] =  pyrol.ParameterList()
    params['General']['Output Level'] = 1
    params['General']['Secant'] = pyrol.ParameterList()
    params['General']['Secant']['Type'] = "Limited-Memory BFGS"
    params['General']['Secant']['Use as Preconditioner'] = False
    params['General']['Secant']['Use as Hessian'] = False
    params['General']['Secant']['Maximum Storage'] = 5
    params['General']['Krylov'] = pyrol.ParameterList()
    params['General']['Krylov']['Iteration Limit'] = 20
    params['Step'] = pyrol.ParameterList()
    params['Step']['Trust Region'] = pyrol.ParameterList()
    params['Step']['Trust Region']['Subproblem Solver'] = 'Truncated CG'
    params['Status Test'] = pyrol.ParameterList()
    params['Status Test']['Iteration Limit'] = 10
    return params


class GotchaRolModel:

    def __init__(self, objective: Union[GotchaObjective, TuckerObjective], initial_decomp) -> None:
        x = vu.ttensor_to_rolvec(initial_decomp, copy=True)
        g = x.dual()
        self.objective = objective
        self._rol_x = x
        self._rol_g = g
        self._rol_objective = None
        self._rol_problem   = None
        self._rol_params    = None
        self._rol_solver    = None
        return
    
    def solve(self, rol_params=None, precondition=True):
        self._rol_objective = gro.GotchaRolObjective(precondition, self.objective)
        self._rol_problem   = pyrol.Problem(self._rol_objective, self._rol_x, self._rol_g)
        self._rol_params = rol_params if rol_params is not None else build_parameter_list()
        self._rol_solver = pyrol.Solver(self._rol_problem, self._rol_params)
        stream = pyrol.getCout()
        self._rol_solver.solve(stream)
        self.decomp = vu.rolvec_to_ttensor(self._rol_x)
        return




class GocchaRolModel:

    def __init__(self, objective: Union[GocchaObjective, CPObjective], initial_decomp) -> None:
        x = vu.ktensor_to_rolvec(initial_decomp, copy=True)
        g = x.dual()
        self.objective = objective
        self._rol_x = x
        self._rol_g = g
        self._rol_objective = None
        self._rol_problem   = None
        self._rol_params    = None
        self._rol_solver    = None
        return

    def solve(self, rol_params=None, precondition=True):
        self._rol_objective = gro.GocchaRolObjective(precondition, self.objective)
        self._rol_problem   = pyrol.Problem(self._rol_objective, self._rol_x, self._rol_g)
        self._rol_params = rol_params if rol_params is not None else build_cp_parameter_list()
        self._rol_solver = pyrol.Solver(self._rol_problem, self._rol_params)
        stream = pyrol.getCout()
        self._rol_solver.solve(stream)
        self.decomp = vu.rolvec_to_ktensor(self._rol_x)
        return

