import numpy as np

import pyrol
from goated.rol_interface.vectors import TuckerVector, CPVector

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


class GotchaRolObjective(pyrol.Objective):

    def __init__(self, precondition, objective):
        super().__init__()
        self._objective = objective
        self._precondition = precondition

    def update(self, x, update_type, iter):
        x = x.to_ttensor()
        self._objective.update(x)

    def value(self, x, tol):
        x = x.to_ttensor()
        return self._objective.value(x)

    def gradient(self, g, x, tol):
        x = x.to_ttensor()
        temp = self._objective.gradient(x)
        temp = TuckerVector.from_ttensor(temp)
        g.set(temp)

    def hessVec(self, hv, v, x, tol):
        x = x.to_ttensor()
        v = v.to_ttensor()
        temp = self._objective.gn_hessvec(x,v)
        temp = TuckerVector.from_ttensor(temp)
        hv.set(temp)

    def precond(self, pv, v, x, tol):
        if not self._precondition:
            pv.set(v)
            return
        x = x.to_ttensor()
        v = v.to_ttensor()
        temp = self._objective.gn_bd_precvec(x,v)
        temp = TuckerVector.from_ttensor(temp)
        pv.set(temp)

    def compute_hessian(self, x, parallel=False):
        n = x.dimension()
        self.update(x,[],[])
        hv = x.clone()
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
        return H
    
    def compute_hessian_and_precond_hessian(self, x, parallel=False):
        n = x.dimension()
        self.update(x,[],[])
        hv = x.clone()
        pv = x.clone()
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
        return H, Hpre


class GocchaRolObjective(pyrol.Objective):

    def __init__(self, precondition, objective):
        super().__init__()
        self._objective = objective
        self._precondition = precondition

    def update(self, x, update_type, iter):
        x = x.to_ktensor()
        self._objective.update(x)

    def value(self, x, tol):
        x = x.to_ktensor()
        return self._objective.value(x)

    def gradient(self, g, x, tol):
        x = x.to_ktensor()
        self._objective.update(x)
        temp = self._objective.gradient(x)
        temp = CPVector.from_ktensor(temp)
        g.set(temp)

    def hessVec(self, hv, v, x, tol):
        x = x.to_ktensor()
        v = v.to_ktensor()
        self._objective.update(x)
        temp = self._objective.hessvec(x,v)
        temp = CPVector.from_ktensor(temp)
        hv.set(temp)

    def precond(self, pv, v, x, tol):
        if not self._precondition:
            pv.set(v)
            return
        x = x.to_ktensor()
        v = v.to_ktensor()
        self._objective.update(x)
        temp = self._objective.precvec(x,v)
        temp = CPVector.from_ktensor(temp)
        pv.set(temp)
