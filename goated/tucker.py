import pyttb as ttb
import numpy as np

import goated.utils.tprod_linops as linops
from collections import defaultdict
import time as _time


class TuckerObjective:

    def __init__(self, X, s=None):
        self.X = X
        self.s = s
        if self.s is None:
            self.s = self.X.norm()**2
        self.times = defaultdict(list)

    def full(self, M):
        # Same as M.full(), except saves the intermediate tensors
        ZZ = M.core.ttm(M.factor_matrices[0],0)
        self.Z = [ ZZ ]
        for i in range(1,M.ndims):
            ZZ = ZZ.ttm(M.factor_matrices[i],i)
            self.Z.append(ZZ)
        return ZZ
        
    def update(self, M):
        self.Mf = self.full(M)

        # compute intermediate tenmats used in gradient() and hessvec
        self.Zt = [ M.core.to_tenmat(np.array([0])).data.T ]
        for i in range(1,M.ndims):
            self.Zt.append(self.Z[i-1].to_tenmat(np.array([i])).data.T)

        # whether we need to recompute point data in hessian/preconditioner
        self.recompute_hess = True
        self.recompute_prec = True
        
    def value(self, M):
        Y = self.Mf-self.X
        F = (Y.norm()**2)/self.s
        return F
    
    def gradient(self, M):
        tic = _time.time()
        Y = (2/self.s)*(self.Mf-self.X)
        Gf = [None]*M.ndims
        Zb = Y
        for i in range(M.ndims-1,-1,-1):
            Gf[i] = Zb.to_tenmat(np.array([i])).data @ self.Zt[i]
            Zb = Zb.ttm(M.factor_matrices[i].T,i)
        G = ttb.ttensor(Zb,Gf)
        toc = _time.time()
        self.times['gradient'].append(toc - tic)
        return G
    
    def gn_hessvec(self, M, V):
        tic = _time.time()
        # Form tangent of reconstructed tensor
        Zd = M.core.ttm(V.factor_matrices[0],0) + V.core.ttm(M.factor_matrices[0],0)
        for i in range(1, M.ndims):
            Zd = self.Z[i-1].ttm(V.factor_matrices[i],i) + Zd.ttm(M.factor_matrices[i],i)

        # compute Gauss-Newton Hessian-vector product
        Zbd = (2/self.s)*Zd
        Hv = [None]*M.ndims
        for i in range(M.ndims-1,-1,-1):
            Hv[i]= Zbd.to_tenmat(np.array([i])).data @ self.Zt[i]
            Zbd = Zbd.ttm(M.factor_matrices[i].T,i)
        Hv = ttb.ttensor(Zbd, Hv)

        self.recompute_hess = False
        toc = _time.time()
        self.times['gn_hessvec'].append(toc - tic)
        return Hv
    
    def recompute_bd_prec(self, M):
        tic = _time.time()
        n = M.ndims
        grams = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(n)]
        self.C = []
        for k in range(n):
            D = np.array([1])
            for i in reversed(range(n)):
                if i != k:
                    D = np.kron(D, grams[i])
            G = M.core.to_tenmat(np.array([k])).data
            D = (2/self.s) * G @ D @ G.T
            block = linops.InvPosDef(D)
            self.C.append(block)
        grams[0] *= (2/self.s)
        kron_args = [linops.InvPosDef(gram) for gram in grams[::-1]]
        B = linops.KronStructured(kron_args)
        self.C.append(B)
        toc = _time.time()
        self.times['recompute_bd_prec'].append(toc - tic)
        return

    def gn_bd_precvec(self, M, V):
        if self.recompute_prec:
            self.recompute_bd_prec(M)
        
        tic = _time.time()
        Pv = []
        for k in range(M.ndims):
            Ck = self.C[k]
            Vk = V.factor_matrices[k]
            try:
                Pvk = Vk @ Ck
            except ValueError:
                Pvk = Ck @ Vk.ravel(order='F')
                Pvk = Pvk.reshape(Vk.shape, order='F')
            Pv.append(Pvk)

        Pvc = V.core.data.ravel(order='F')
        Pvc = self.C[-1] @ Pvc
        Pvc = Pvc.reshape(V.core.shape, order='F')
        Pv  = ttb.ttensor(ttb.tensor(Pvc), Pv)
        self.recompute_prec = False
        toc = _time.time()
        self.times['gn_bd_precvec'].append(toc - tic)
        return Pv


class GotchaObjective(TuckerObjective):

    def __init__(self, X, scaler, goals, weights, a, b, jacobi=True):
        super().__init__(X, s=1.0)
        self.scaler = scaler
        self.goals = goals
        self.weights = weights
        self.a = a
        self.b = b
        self.jacobi = jacobi
        self.block_jacobi_ops_cache = []
        
    def update(self, M):
        super().update(M)
        self.Mfs = self.scaler.unscale_tensor(self.Mf)
        
    def value(self, M):
        # Tensor term
        F = self.a*super().value(M)

        # Goal terms
        for w,g in zip(self.weights,self.goals):
            F += (self.b * w) * g.computeValue(self.Mfs)
        return F
    
    def gradient(self, M):
        tic = _time.time()
        # Tensor term
        Y = self.Mf-self.X
        
        # Goal terms
        Yg = np.zeros(M.shape)
        for w,g in zip(self.weights,self.goals):
            Yg += w * g.computeDeriv(self.Mfs)
        Yg = ttb.tensor(Yg)
        Yg = self.scaler.unscale_tensor(Yg, shift=False)

        # Compute gradiennt
        Gf = [None]*M.ndims
        Zb = (2*self.a)*Y + self.b*Yg
        for i in range(M.ndims-1,-1,-1):
            Gf[i] = Zb.to_tenmat(np.array([i])).data @ self.Zt[i]
            Zb = Zb.ttm(M.factor_matrices[i].T,i)
        G = ttb.ttensor(Zb,Gf)
        toc = _time.time()
        self.times['gradient'].append(toc - tic)
        return G
    
    def gn_hessvec(self, M, V):
        tic = _time.time()
        # Form tangent of reconstructed tensor
        Zd = M.core.ttm(V.factor_matrices[0],0) + V.core.ttm(M.factor_matrices[0],0)
        for i in range(1, M.ndims):
            Zd = self.Z[i-1].ttm(V.factor_matrices[i],i) + Zd.ttm(M.factor_matrices[i],i)
        Md = ttb.tensor(Zd.data)
        Md = self.scaler.unscale_tensor(Md, shift=False)
        Md = Md.data

        recomp_times = []

        # compute necessary gradient info
        Yd = np.zeros(M.shape,order='F')
        for w,g in zip(self.weights,self.goals):
            var = g.var
            time = g.time
            if not isinstance(time, np.ndarray):
                time = np.array(time)
            if not isinstance(var, np.ndarray):
                var = np.array(var)
            num_time = len(time)
            if self.recompute_hess:
                # TODO: see how this contributes to the runtime of gn_hessvec.
                ticc = _time.time()
                val,jac = g.computeTarget(self.Mfs, compute_deriv=True)
                tocc = _time.time()
                recomp_times.append(tocc - ticc)
                setattr(g,'val',val)
                setattr(g,'jac',jac)
            else:
                val = g.val
                jac = g.jac

            DEBUG = False
                
            # compute val_dot (could tangent-differentiate fcn, but since we already have jac, we just do a mat-vec)
            jact = jac[:,:,:,time]
            Mdt  = Md[:,:,:,time]
            val_dot = np.zeros((num_time,))
            val_dot[:] = np.einsum('hijk,hijk->k', jact[:,:,var,:], Mdt[:,:,var,:])
            if DEBUG:
                vd = np.zeros((num_time,))
                for i in range(num_time):
                    vd[i] = np.reshape(jac[:,:,var,time[i]],(1,-1),order='F') @ np.reshape(Md[:,:,var,time[i]],(-1,1),order='F')
                
            # compute dot gradient tensor dF/dM(i,j,v,t)
            jac_dot = np.zeros(jac.shape)
            mask_t = np.zeros(jac.shape, dtype=bool)
            mask_v = np.zeros(jac.shape, dtype=bool)
            mask_t[:,:,:,time] = True
            mask_v[:,:,var,:]  = True 
            ro = 'C'  # doesn't matter, just be explicit.
            mask = (mask_t & mask_v).ravel(order=ro)
            jac_dot.ravel(order=ro)[mask] = (2*val_dot[None,None,:]*jact[:,:,var,:]).ravel(order=ro)
            if DEBUG:
                jd = np.zeros(jac.shape)
                for i in range(num_time):
                    jd[:,:,var,time[i]] = 2*val_dot[i]*jac[:,:,var,time[i]]

            Yd += w*jac_dot
        Yd = self.scaler.unscale_tensor(Yd, shift=False)

        # compute Gauss-Newton Hessian-vector product
        Zbd = (2*self.a)*Zd + self.b*Yd
        Hv = [None]*M.ndims
        for i in range(M.ndims-1,-1,-1):
            Hv[i] = Zbd.to_tenmat(np.array([i])).data @ self.Zt[i]
            Zbd = Zbd.ttm(M.factor_matrices[i].T,i)
        Hv = ttb.ttensor(Zbd, Hv)

        self.recompute_hess = False
        toc = _time.time()
        recomp_hess_time = sum(recomp_times)
        self.times['recompute hessian'].extend(recomp_times)
        self.times['gn_hessvec, marginal'].append(toc - tic - recomp_hess_time)
        return Hv

    def gn_diag_block_goal_updates(self, M):
        tic = _time.time()
        factor_cols = [[] for _ in range(M.ndims + 1)]
        for w, g in zip(self.weights, self.goals):
            goal_scale = np.sqrt(2*self.b*w)
            time = g.time
            _, jac = g.computeTarget(self.Mfs, compute_deriv=True)
            jac_mat_shape = M.shape[0:M.ndims-1] + (1,)
            for t in time:
                jac_t = ttb.tensor(jac[:,:,:,t], shape=jac_mat_shape, copy=False)
                jac_t = self.scaler.unscale_tensor(jac_t, shift=False)

                for n in range(M.ndims):
                    mats, dims = [], []
                    for i in range(M.ndims):
                        if i != n:
                            if i == M.ndims-1:
                                mats.append(np.reshape(M.factor_matrices[i][t,:],(1,-1)))
                            else:
                                mats.append(M.factor_matrices[i])
                            dims.append(i)
                    D = jac_t.ttm(mats, dims=dims, transpose=True)
                    D = D.to_tenmat(np.array([n])).data @ M.core.to_tenmat(np.array([n])).data.T
                    if n == M.ndims-1:
                        D2 = np.zeros((M.shape[n], M.core.shape[n]))
                        D2[t,:] = D
                        D = D2
                    D = goal_scale * np.reshape(D, (-1,1), order='F')
                    factor_cols[n].append(D)

                # Core term
                mats = [M.factor_matrices[i] for i in range(M.ndims-1)]
                mats.append(np.reshape(M.factor_matrices[-1][t,:],(1,-1)))
                dims = list(range(M.ndims))
                D = jac_t.ttm(mats, dims=dims, transpose=True)
                D = goal_scale * np.reshape(D.data, (-1,1), order='F')
                factor_cols[-1].append(D)

        factors = [np.column_stack(cols) for cols in factor_cols]
        toc = _time.time()
        self.times['gn_diag_block_goal_updates'].append(toc - tic)
        return factors

    # Computes the diagonal blocks of the Gauss-Newton Hessian with respect to the factor matrices
    def compute_diag_blocks(self, M):
        S = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(M.ndims)]
        C = []
        goal_updates = self.gn_diag_block_goal_updates(M)

        for n in range(M.ndims):
            D = np.array([1])
            for i in range(M.ndims-1, -1, -1):
                if i != n:
                    D = np.kron(D, S[i])
            G = M.core.to_tenmat(np.array([n])).data
            D = (2*self.a) * G @ D @ G.T
            D = np.kron(D, np.eye(M.shape[n]))
            D += goal_updates[n] @ goal_updates[n].T
            C.append(D)

        D = np.array([1])
        for i in range(M.ndims-1, -1, -1):
            D = np.kron(D, S[i])
        D = (2*self.a) * D
        D += goal_updates[-1] @ goal_updates[-1].T
        C.append(D)

        return C
    
    def recompute_bd_prec(self, M):
        if self.jacobi:
            self.recompute_bj_prec(M)
        else:
            TuckerObjective.recompute_bd_prec(self, M)
        return

    def recompute_bj_prec(self, M):
        goals = self.gn_diag_block_goal_updates(M)
        tic = _time.time()
        n = M.ndims
        grams = [M.factor_matrices[k].T @ M.factor_matrices[k] for k in range(n)]
        self.C = []
        for k in range(n):
            D = np.array([1])
            for i in reversed(range(n)):
                if i != k:
                    D = np.kron(D, grams[i])
            G = M.core.to_tenmat(np.array([k])).data
            D = (2*self.a) * G @ D @ G.T
            block = linops.InvUpdatedKronPosDef([D, np.eye(M.shape[k])], goals[k])
            self.C.append(block)
        grams[0] *= (2*self.a)
        kron_args = grams[::-1]
        B = linops.InvUpdatedKronPosDef(kron_args, goals[-1])
        self.C.append(B)
        toc = _time.time()
        self.times['recompute_bj_prec, marginal'].append(toc - tic)
        self.block_jacobi_ops_cache.append([op for op in self.C])
        return
