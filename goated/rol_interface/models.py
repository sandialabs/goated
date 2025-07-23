import pyrol

from goated.rol_interface.objectives import GoatedRolObjective
from goated.rol_interface.vectors import TuckerVector, CPVector
from goated.tucker import TuckerObjective
from goated.cp import CPObjective
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
    params['Status Test']['Iteration Limit'] = 9
    return params


class GoatedRolModel:

    def __init__(self, objective: TuckerObjective | CPObjective, initial_decomp) -> None:

        if isinstance(objective, TuckerObjective):
            self._rolvector_type = TuckerVector
        elif isinstance(objective, CPObjective):
            self._rolvector_type = CPVector
        else:
            raise ValueError()

        x = self._rolvector_type.from_tensor(initial_decomp, copy=True)
        g = x.dual()
        self.objective = objective
        self.decomp = None
        self._rol_x = x
        self._rol_g = g
        self._rol_objective = None
        self._rol_problem   = None
        self._rol_params    = None
        self._rol_solver    = None
        return
    
    def default_rol_params(self):
        if issubclass(self._rolvector_type, TuckerVector):
            return build_parameter_list()
        else:
            return build_cp_parameter_list()
    
    def solve(self, rol_params=None, precondition=True):
        self._rol_params    = rol_params if rol_params is not None else self.default_rol_params()
        self._rol_objective = GoatedRolObjective(self.objective, precondition=precondition)
        self._rol_problem   = pyrol.Problem(self._rol_objective, self._rol_x, self._rol_g)
        self._rol_solver    = pyrol.Solver(self._rol_problem, self._rol_params)
        stream = pyrol.getCout()
        self._rol_solver.solve(stream)
        self.decomp = self._rol_x.to_tensor()
        return
