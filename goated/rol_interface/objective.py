import numpy as np

import pyrol
import goated.rol_interface.vectors as uvec

from tqdm import tqdm
import dask
dask.config.set(scheduler='threads', num_workers=4)
from dask.diagnostics import ProgressBar
import pyttb as ttb


class GotchaRolObjective(pyrol.Objective):

    def __init__(self, precondition, objective):
        super().__init__()
        self._objective = objective
        self._precondition = precondition

    def update(self, x, update_type, iter):
        x = uvec.rolvec_to_ttensor(x)
        self._objective.update(x)

    def value(self, x, tol):
        x = uvec.rolvec_to_ttensor(x)
        return self._objective.value(x)

    def gradient(self, g, x, tol):
        x = uvec.rolvec_to_ttensor(x)
        temp = self._objective.gradient(x)
        temp = uvec.ttensor_to_rolvec(temp)
        g.set(temp)

    def hessVec(self, hv, v, x, tol):
        x = uvec.rolvec_to_ttensor(x)
        v = uvec.rolvec_to_ttensor(v)
        temp = self._objective.gn_hessvec(x,v)
        temp = uvec.ttensor_to_rolvec(temp)
        hv.set(temp)

    def precond(self, pv, v, x, tol):
        if not self._precondition:
            pv.set(v)
            return
        x = uvec.rolvec_to_ttensor(x)
        v = uvec.rolvec_to_ttensor(v)
        temp = self._objective.gn_bd_precvec(x,v)
        temp = uvec.ttensor_to_rolvec(temp)
        pv.set(temp)

    def compute_hessian(self, x, parallel=False):
        n = x.dimension()
        self.update(x,[],[])
        hv = x.clone()
        def run(i):
            v = x.basis(i)
            self.hessVec(hv,v,x,0.0)
            out = uvec.trolvec_to_array(hv)
            return out
        if parallel:
            lazy_results = []
            for i in range(n):
                lazy_data = dask.delayed(run)(i)
                lazy_results.append(lazy_data)
            with ProgressBar():
                results = dask.compute(*lazy_results)
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
            out1 = uvec.trolvec_to_array(hv)
            self.precond(pv,hv,x,0.0)
            out2 = uvec.trolvec_to_array(pv)
            return (out1, out2)
        if parallel:
            lazy_results = []
            for i in range(n):
                lazy_data = dask.delayed(run)(i)
                lazy_results.append(lazy_data)
            with ProgressBar():
                results = dask.compute(*lazy_results)
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
        x = uvec.rolvec_to_ktensor(x)
        self._objective.update(x)

    def value(self, x, tol):
        x = uvec.rolvec_to_ktensor(x)
        return self._objective.value(x)

    def gradient(self, g, x, tol):
        x = ttb.ktensor(x.data)
        self._objective.update(x)
        temp = self._objective.gradient(x)
        temp = uvec.ktensor_to_rolvec(temp)
        g.set(temp)

    def hessVec(self, hv, v, x, tol):
        x = ttb.ktensor(x.data)
        v = ttb.ktensor(v.data)
        self._objective.update(x)
        temp = self._objective.hessvec(x,v)
        temp = uvec.ktensor_to_rolvec(temp)
        hv.set(temp)

    def precond(self, pv, v, x, tol):
        if not self._precondition:
            pv.set(v)
            return
        x = ttb.ktensor(x.data)
        v = ttb.ktensor(v.data)
        self._objective.update(x)
        temp = self._objective.precvec(x,v)
        temp = uvec.ktensor_to_rolvec(temp)
        pv.set(temp)
