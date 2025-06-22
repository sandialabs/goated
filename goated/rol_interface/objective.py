import numpy as np

import pyrol
import goated.utils.vectorization as uvec
from goated.rol_interface.FactorVector import FactorVector

from tqdm import tqdm
import dask
dask.config.set(scheduler='threads', num_workers=4)
from dask.diagnostics import ProgressBar
import pyttb as ttb


class GotchaRolObjective(pyrol.Objective):

    def __init__(self, precondition, gotcha):
        super().__init__()
        self._gotcha = gotcha
        self._precondition = precondition

    def update(self, x, update_type, iter):
        x = uvec.vec_to_ttensor(x)
        self._gotcha.update(x)

    def value(self, x, tol):
        x = uvec.vec_to_ttensor(x)
        return self._gotcha.value(x)

    def gradient(self, g, x, tol):
        x = uvec.vec_to_ttensor(x)
        temp = self._gotcha.gradient(x)
        temp = uvec.ttensor_to_vec(temp)
        g.set(temp)

    def hessVec(self, hv, v, x, tol):
        x = uvec.vec_to_ttensor(x)
        v = uvec.vec_to_ttensor(v)
        temp = self._gotcha.gn_hessvec(x,v)
        temp = uvec.ttensor_to_vec(temp)
        hv.set(temp)

    def precond(self, pv, v, x, tol):
        if not self._precondition:
            pv.set(v)
            return
        x = uvec.vec_to_ttensor(x)
        v = uvec.vec_to_ttensor(v)
        temp = self._gotcha.gn_bd_precvec(x,v)
        temp = uvec.ttensor_to_vec(temp)
        pv.set(temp)

    def compute_hessian(self, x, parallel=False):
        n = x.dimension()
        self.update(x,[],[])
        hv = x.clone()
        def run(i):
            v = x.basis(i)
            self.hessVec(hv,v,x,0.0)
            out = uvec.vec_to_array(hv)
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
            out1 = uvec.vec_to_array(hv)
            self.precond(pv,hv,x,0.0)
            out2 = uvec.vec_to_array(pv)
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


"""
TODO: figure out if this is needed.
I think we can rely on GotchaRolObjective.
"""
class GOCP(pyrol.Objective):

    def __init__(self, precondition, cp):
        super().__init__()
        self._cp = cp
        self._precondition = precondition

    def value(self, x, tol):
        x = ttb.ktensor(x.data)
        self._cp.update(x)
        return self._cp.value(x)

    def gradient(self, g, x, tol):
        x = ttb.ktensor(x.data)
        self._cp.update(x)
        temp = self._cp.gradient(x)
        temp = FactorVector(temp)
        g.set(temp)

    def hessVec(self, hv, v, x, tol):
        x = ttb.ktensor(x.data)
        v = ttb.ktensor(v.data)
        self._cp.update(x)
        temp = self._cp.hessvec(x,v)
        temp = FactorVector(temp)
        hv.set(temp)

    def precond(self, pv, v, x, tol):
        if not self._precondition:
            pv.set(v)
            return
        x = ttb.ktensor(x.data)
        v = ttb.ktensor(v.data)
        self._cp.update(x)
        temp = self._cp.precvec(x,v)
        temp = FactorVector(temp)
        pv.set(temp)
