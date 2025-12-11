import numpy as np
import pydoc


from goated.rol_interface.vectors import TuckerVector, CPVector, vector_copy, vector_distance
from goated.cp import CPObjective
from goated.tucker import TuckerObjective, GotchaObjective
from typing import Optional, TypeAlias
import inspect as ins

from pyrol.getTypeName import getTypeName2 as pyrol_types
UpdateType : TypeAlias = pyrol_types('UpdateType') # type: ignore
Objective  : TypeAlias = pyrol_types('Objective')  # type: ignore
Vector     : TypeAlias = pyrol_types('Vector')     # type: ignore

from tqdm import tqdm


def dask_parallel_eval(callable, iterable, num_workers=4):
    import dask
    dask.config.set(scheduler='threads', num_workers=num_workers)
    from dask.diagnostics import ProgressBar
    lazy_results = []
    for i in iterable:
        lazy_data = dask.delayed(callable)(i)
        lazy_results.append(lazy_data)
    with ProgressBar():
        results = dask.compute(*lazy_results)
    return results


class TrustRegionObjective(Objective):
    """
    pyrol.Objective is an alias for pyrol.pyrol.ROL.Objective_FSsolver_double_t
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.debug = kwargs.get('debug', True)
        self._last_x    : Optional[Vector]     = None
        self._last_ut   : Optional[UpdateType] = None
        self._last_iter : int = 0

    def debug_check_x_unchanged(self, x: Vector) -> bool:
        if self.debug:
            return (self._last_x is None) or (vector_distance(self._last_x, x) == 0)
        return True
    
    def update(self, x, ut : UpdateType, iter: int) -> None:
        self._last_x    = vector_copy(x)
        self._last_ut   = ut
        self._last_iter = iter

    def value(self, x, tol) -> float:
        assert(self.debug_check_x_unchanged(x))
        return 0.0

    def gradient(self, g, x, tol) -> None:
        assert(self.debug_check_x_unchanged(x))

    def hessVec(self, hv, v, x, tol) -> None:
        assert(self.debug_check_x_unchanged(x))

    def precond(self, pv, v, x, tol) -> None:
        assert(self.debug_check_x_unchanged(x))


class GoatedRolObjective(TrustRegionObjective):

    def __init__(self, objective : TuckerObjective | CPObjective, precondition: bool, debug: bool=True):
        if isinstance(objective, TuckerObjective):
            self._rolvector_type = TuckerVector
        elif isinstance(objective, CPObjective):
            self._rolvector_type = CPVector
        else:
            raise ValueError()
        self._our_objective = objective
        self._precondition  = precondition
        TrustRegionObjective.__init__(self, debug=debug)

    def update(self, x, ut : UpdateType, iter: int):
        x_ten = x.to_tensor()
        update_grad = ut != UpdateType.Trial
        update_prec = update_grad and self._precondition
        self._our_objective.update(
            x_ten, prec=update_prec, grad=update_grad
        )
        TrustRegionObjective.update(self, x, ut, iter)

    def value(self, x, tol):
        TrustRegionObjective.value(self, x, tol)
        return self._our_objective.value()

    def gradient(self, g, x, tol):
        TrustRegionObjective.gradient(self, g, x, tol)
        temp_ten = self._our_objective.gradient()
        temp_vec = self._rolvector_type.from_tensor(temp_ten)
        g.set(temp_vec)

    def hessVec(self, hv, v, x, tol):
        TrustRegionObjective.hessVec(self, hv, v, x, tol)
        v_ten = v.to_tensor()
        temp_ten = self._our_objective.hessvec(v_ten)
        temp_vec = self._rolvector_type.from_tensor(temp_ten)
        hv.set(temp_vec)

    def precond(self, pv, v, x, tol):
        TrustRegionObjective.precond(self, pv, v, x, tol)
        if not self._precondition:
            pv.set(v)
            return
        v_ten = v.to_tensor()
        temp_ten = self._our_objective.precvec(v_ten)
        temp_vec = self._rolvector_type.from_tensor(temp_ten)
        pv.set(temp_vec)

    def compute_hessian(self, x, parallel=False):
        n = x.dimension()
        hv = x.clone()
        last_x = vector_copy(self._last_x)
        self.update(x, self._last_ut, self._last_iter)
        def run(i):
            v = x.basis(i)
            self.hessVec(hv,v,x,0.0)
            out = hv.to_numpy_1d()
            return out
        if parallel:
            results = dask_parallel_eval(run, range(n))
        else:
            results = [run(i) for i in tqdm(range(n))]
        H = np.zeros((n,n))
        for i,v in enumerate(results):
            H[:,i] = v
        self.update(last_x, self._last_ut, self._last_iter)
        return H
    
    def compute_hessian_and_precond_hessian(self, x, parallel=False):
        n = x.dimension()
        hv = x.clone()
        pv = x.clone()
        last_x = vector_copy(self._last_x)
        self.update(x, self._last_ut, self._last_iter)
        def run(i):
            v = x.basis(i)
            self.hessVec(hv,v,x,0.0)
            out1 = hv.to_numpy_1d()
            self.precond(pv,hv,x,0.0)
            out2 = pv.to_numpy_1d()
            return (out1, out2)
        if parallel:
            results = dask_parallel_eval(run, range(n))
        else:
            results = [run(i) for i in tqdm(range(n))]
        H = np.zeros((n,n))
        Hpre = np.zeros((n,n))
        for i,(u,v) in enumerate(results):
            H[:,i]    = u
            Hpre[:,i] = v
        self.update(last_x, self._last_ut, self._last_iter)
        return H, Hpre
