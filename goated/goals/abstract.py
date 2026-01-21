import numpy as np
from numpy.typing import ArrayLike
from pyttb import tensor, ttensor, ktensor  # type: ignore
from typing import TypeAlias, Tuple, List, Optional, Sequence, Any
from abc import ABC, abstractmethod

Tensor : TypeAlias = tensor | ttensor | ktensor


class Goal(ABC):
    """
    An abstract nonlinear least squares term in the objective of a
    minimization problem over tensors.
    
    Concrete child classes must implement computeVector and adjoint_jvp.
    We describe the semantics of these abstract functions in terms of a
    hypothetical object G = Goal(ground_truth, **kwargs).

    Bare basics
    -----------
    For a Tensor U of shape G.domain_shape, the following expressions
    are well-formed: 

        (vec, jac) = G.computeVector( U, compute_deriv=True )
        s          = G.computeScalar( U )
        dsdU       = G.adjoint_jvp( jac, vec - G.target  )

    The returned values should have the following properties:

      (1) s is the squared Frobenius norm of vec - G.target.
      (2) dsdU is the gradient of G.computeScalar at U.

    Conceptually, jac should be some representation of the Jacobian
    of the map from U to vec. G.adjoint_jvp should be a bilinear 
    function that implements the matrix-vector product between the
    adjoint of the given Jacobian and the given vector.

    NOTE: The instance variable G.target is set automatically during
    the call to Goal.__init__(self, ground_truth ).

    Being efficient
    ---------------
    For a Tensor U of shape G.domain_shape, the expression

        (vec, jac) = G.computeVector( U, compute_deriv=False )

    must be well-formed. However, in this case, no other requirements
    are placed on jac (it's valid to return jac==None).

    The object G has instance variables G.cached_jac and G.cached_vec
    where calling code can choose to save results from a previous
    call to G.computeVector(*, compute_deriv=True ).
    """

    def __init__(self, ground_truth : Tensor, **kwargs) -> None:
        self.domain_shape = ground_truth.shape
        self.cached_vec  : np.ndarray = np.empty(())
        self.cached_jac  : Optional[Any] = None
        self.target, _ = self.computeVector(ground_truth, compute_deriv=False)
        return
    
    @abstractmethod
    def computeVector(self, U : Tensor, compute_deriv=False) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError()
    
    @abstractmethod
    def adjoint_jvp(self, jac: Any, vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def computeScalar(self, U : Tensor) -> np.floating:
        vec, _ = self.computeVector(U, compute_deriv=False)
        diff = vec - self.target
        F = np.linalg.norm(diff)**2
        return F

    def computeGrad(self, U : Tensor):
        vec, jac = self.computeVector(U, compute_deriv=True)
        return self.adjoint_jvp(jac, vec)


class TimeSeparableGoal(Goal):
    """
    Let G = TimeSeparableGoal( ground_truth, var, time ), and let U 
    be a Tensor of shape G.domain_shape. If

        (vec, jac) = G.computeVector( U, compute_deriv=True )

    then vec.shape == (G.time.size,) and jac.shape == G.domain_shape.

    The map from U to vec must be separable; vec[i] can only depend on
    U[ ..., G.var, G.time[i] ]. This limited dependence makes it possible to
    pack the entire Jacobian into jac, which is a far smaller array
    than a naively-stored Jacobian. See the docstring of computeVector
    for a full specification of jac.
    """

    def __init__(self, ground_truth : Tensor, var: ArrayLike, time: ArrayLike):
        if not isinstance(var, np.ndarray):
            var = np.array(var)
        if not isinstance(time, np.ndarray):
            time = np.array(time)
        self.var  : np.ndarray = var
        self.time : np.ndarray = time
        super().__init__(ground_truth)
        # ^ computes self.domain_shape and self.target.
        assert np.all((0 <= self.var)  & (self.var  < self.domain_shape[-2]))
        assert np.all((0 <= self.time) & (self.time < self.domain_shape[-1]))

        self.cached_jac : np.ndarray = np.empty(())
        index_vecs = [np.arange(d) for d in self.domain_shape[:-2]]
        index_vecs.append(self.var)
        index_vecs.append(self.time)
        self._nonconst_indices : tuple[np.ndarray,...] = np.ix_(*index_vecs)
        # ^ If someone calls `self.computeVector` with a Tensor `U` and no error
        #   is raised, then the expression `U[self._nonconst_indices]` is well-formed,
        #   it evaluates to a Tensor, and the value returned from `self.computeVector`
        #   will only depend on `U` by way of `U[self._nonconst_indices]`.
        #
        #   ...
        # 
        #   If you're wondering what np.ix_ actually does, the basic idea can be seen
        #   when called with two arguments. Given row indices `I`, column indices `J`,
        #   and an ndarray `X`, we'll have elementwise equality
        #     
        #       X[np.ix_(I, J)] == np.array([[X[i,j] for j in J] for i in I]).
        #     
        #   Put another way, np.ix_(I, J) returns an object that can be used for
        #   selecting a sub-array given by the Cartesian product of `I` and `J`.
        #   This idea generalizes easily to three or more dimensions with calls
        #   like `np.ix_(I, J, K)`, etc ...
        #   
        self.DEBUG = False

    def computeVector(self, U : Tensor, compute_deriv=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return a pair of ndarrays, (vec, compact_jac), satisfying the conditions
        described below in terms of U_full = U.double() (the numpy ndarray
        representation of U).
        
          (1) vec.ndim == 1 and vec.size == self.time.size.

          (2) vec only depends on U_full[ ..., self.var, self.time], and
              the dependence is separable with respect to time. Specifically,
              vec[i] only depends on U_full[ ..., self.var, self.time[i] ].
        
          (3) If compute_deriv is True, then compact_jac.shape == self.domain_shape
              and compact_jac[ ..., self.time[i] ] is the derivative of vec[i] with
              respect to U_full[ ..., self.time[i] ].

        """
        raise NotImplementedError()

    def adjoint_jvp(self, compact_jac: np.ndarray, v: np.ndarray):    
        grad = compact_jac.copy()
        assert grad.shape == self.domain_shape
        temp = 2*np.reshape(v, self._nonconst_indices[-1].shape)
        grad[self._nonconst_indices] *= temp 
        # ^ left-hand and right-hand sides have different shapes,
        #   but the multiplication is resolved by broadcasting temp.
        #
        return grad

    def computeGrad(self, U : Tensor, use_cached: bool):
        if use_cached:
            vec, jac = self.cached_vec, self.cached_jac
        else:
            vec, jac = self.computeVector(U, compute_deriv=True)
        diff = vec - self.target
        return self.adjoint_jvp(jac, diff)

    def _gn_hessvec(self, Md) -> np.ndarray:
        # TODO: extend this to handle arbitrarily-many "spatial" dimensions.
        i3 = self.var
        i4 = self.time
        jac = self.cached_jac
        jact = jac[ :, :, :, i4 ]
        Mdt  =  Md[ :, :, :, i4 ]
        val_dot = np.zeros((i4.size,))
        val_dot[:] = np.einsum('hijk,hijk->k', jact[:,:,i3,:], Mdt[:,:,i3,:])
        if self.DEBUG:
            vd = np.zeros((i4.size,))
            for i in range(i4.size):
                vd[i] = np.reshape(jac[:,:,i3,i4[i]],(1,-1),order='F') @ np.reshape(Md[:,:,i3,i4[i]],(-1,1),order='F')
            assert np.linalg.norm(vd - val_dot) <= 1e-8 * np.maximum(1.0, np.linalg.norm(vd))

        # compute dot gradient tensor dF/dM(i,j,v,t)
        jac_dot = np.zeros(jac.shape)
        mask_t  = np.zeros(jac.shape, dtype=bool)
        mask_v  = np.zeros(jac.shape, dtype=bool)
        mask_t[:,:,:,i4] = True
        mask_v[:,:,i3,:]  = True 
        ro = 'C'  # doesn't matter, just be explicit.
        mask = (mask_t & mask_v).ravel(order=ro)
        jac_dot.ravel(order=ro)[mask] = (2*val_dot[None,None,:]*jact[:,:,i3,:]).ravel(order=ro)
        if self.DEBUG:
            jd = np.zeros(jac.shape)
            for i in range(i4.size):
                jd[:,:,i3,i4[i]] = 2*val_dot[i]*jac[:,:,i3,i4[i]]
            assert np.linalg.norm(jac_dot - jd) <= 1e-8 * np.maximum(1.0, np.linalg.norm(jd))
        return jac_dot

    def _tucker_gn_block_diag_goal_updates(self, w: float, M: ttensor, scaler, factor_cols: list[list]):
        # TODO: extend this to handle arbitrarily-many "spatial" dimensions.
        scale = np.sqrt(2 * w)
        jac   = self.cached_jac
        shape = self.domain_shape
        jac_mat_shape = shape[:-1] + (1,)
        ndim = len(shape)
        i4 = self.time
        for t in i4:
            jac_t = tensor(jac[:,:,:,t], shape=jac_mat_shape, copy=False)
            jac_t = scaler.unscale_tensor(jac_t, shift=False)
            for n in range(ndim):
                mats, dims = [], []
                for i in range(ndim):
                    if i != n:
                        if i == ndim-1:
                            mats.append(np.reshape(M.factor_matrices[i][t,:],(1,-1)))
                        else:
                            mats.append(M.factor_matrices[i])
                        dims.append(i)
                D = jac_t.ttm(mats, dims=dims, transpose=True)
                D = D.to_tenmat(np.array([n])).data @ M.core.to_tenmat(np.array([n])).data.T
                if n == ndim-1:
                    D2 = np.zeros((shape[n], M.core.shape[n]))
                    D2[t,:] = D
                    D = D2
                D = scale * np.reshape(D, (-1,1), order='F')
                factor_cols[n].append(D)

            # Core term
            mats = [M.factor_matrices[i] for i in range(ndim-1)]
            mats.append(np.reshape(M.factor_matrices[-1][t,:],(1,-1)))
            dims = list(range(ndim))
            D = jac_t.ttm(mats, dims=dims, transpose=True)
            D = scale * np.reshape(D.data, (-1,1), order='F')
            factor_cols[-1].append(D)
        return


class Goals:
    """
    Constituent TimeSeparableGoal objects define their targets in terms 
    of the un-scaled tensor, but GocchaObjective works in terms of
    a scaled tensor. So this class maintains its own scaler.
    """

    def __init__(self, scaler, goals : List[TimeSeparableGoal], weights : Optional[Sequence[float] | np.ndarray] = None):
        assert len(goals) > 0
        self.scaler = scaler
        self.goals = goals
        if weights is None:
            weights = np.ones((len(goals),), dtype=float)
        elif not isinstance(weights, np.ndarray):
            weights = np.array(weights)
        self.weights : np.ndarray = weights
        self.M = None
        self.Mfs : tensor  = tensor()
        self._shape : Tuple[int,...] = goals[0].domain_shape
        self._ndim  : int = len(self._shape)
        self._grad  : tensor = tensor()
        return
        
    def update(self, M, Mf: tensor, grad=True, jacs=True):
        self.M = M  # not used in this base class
        self.Mfs = self.scaler.unscale_tensor(Mf)
        if grad or jacs:
            self.recompute_jacs()
        if grad:
            self.recompute_grad(use_cached_jacs=True)
        return

    def recompute_jacs(self) -> None:
        for _,g in zip(self.weights,self.goals):
            val, jac = g.computeVector(self.Mfs, compute_deriv=True)
            g.cached_vec = val
            g.cached_jac = jac
        return

    def value(self) -> float:
        F = 0
        for w,g in zip(self.weights, self.goals):
            F += w * g.computeScalar(self.Mfs)
        return F

    def recompute_grad(self, use_cached_jacs: bool) -> None:
        Yg = np.zeros(self._shape)
        for w,g in zip(self.weights, self.goals):
            Yg += w * g.computeGrad(self.Mfs, use_cached_jacs)
        Yg = tensor(Yg)
        Yg = self.scaler.unscale_tensor(Yg, shift=False)
        # ^ That's there because of the chain rule; we applied
        #   unscale_tensor to get our hands on Mfs in the first
        #   place.
        self._grad : tensor = Yg
        return

    def gradient_wrt_reconstruction(self):
        return self._grad

    def hessvec_wrt_reconstruction(self, Md: tensor) -> tensor:
        Md = self.scaler.unscale_tensor(Md, shift=False).data
        Yd = np.zeros(self._shape, order='F')
        for w,g in zip(self.weights, self.goals):
            jac_dot = g._gn_hessvec(Md)
            Yd += w*jac_dot
        Yd = self.scaler.unscale_tensor(Yd, shift=False)
        # ^ That's there because of the chain rule; we applied
        #   unscale_tensor to Md.
        return Yd

    def eval_goals(self, U: tensor, scaled=False):   
        if scaled:
            U = self.scaler.unscale_tensor(U)
        v = np.array([g.computeScalar(U) for g in self.goals])
        return v 

    def auto_reweight(self, U: tensor, scaled=False):
        v = self.eval_goals(U, scaled=scaled)
        self.weights = 1 / (v * len(self.goals))
        return


class CPGoals(Goals):

    def __init__(self, scaler, goals: List[TimeSeparableGoal], weights: Sequence[float] | np.ndarray | None = None):
        super().__init__(scaler, goals, weights)
        self.M : ktensor = ktensor()

    def update(self, M: ktensor, Mf: tensor, grad=True, jacs=True):
        super().update(M, Mf, grad, jacs)
        return


class TuckerGoals(Goals):

    def __init__(self, scaler, goals: List[TimeSeparableGoal], weights: Sequence[float] | np.ndarray | None = None):
        super().__init__(scaler, goals, weights)
        self.M : ttensor = ttensor()
        return
        
    def update(self, M: ttensor, Mf: tensor, grad=True, jacs=True):
        super().update(M, Mf, grad, jacs)
        return

    def tucker_gn_diag_block_goal_updates(self) -> list[np.ndarray]:
        factor_cols = [[] for _ in range(self._ndim + 1)]
        for w, g in zip(self.weights, self.goals):
            g._tucker_gn_block_diag_goal_updates(w, self.M, self.scaler, factor_cols)
        factors = [np.column_stack(cols) for cols in factor_cols]
        return factors
