import numpy as np
import matplotlib.pyplot as plt
from goated.goals.abstract import PhysicsGoal
from goated.utils.exo import ExoInfo
from typing import Tuple


def set_docstring(docstr):
    def assign(fn):
        fn.__doc__ = docstr
        return fn
    return assign


DOCSTRING_TEMPLATE_COMPUTE = \
"""
Compute COMPUTE_SUB.

Parameters
----------
X : ttb.ktensor or ndarray of shape (nx, ny[, nz], nvar, ntime)
    Full or reduced data tensor.

var : sequence of int
    VAR_SUB in the variable-mode of X.

time : sequence of int
    Time-step indices at which to evaluate the integral.

exo : ExoInfo
    Provides quadrature weights and mappings for spatial integration.

compute_deriv : bool, default=False
    If True, also compute derivative w.r.t. X.

Returns
-------
p : ndarray of length len(time)
    OUTPUT_SUB

jac : ndarray or empty
    If compute_deriv is True, then this is the Jacobian ∂p/∂X of shape
    (nx,ny[,nz],nvar,ntime); otherwise an empty array.
"""


@set_docstring(
    DOCSTRING_TEMPLATE_COMPUTE\
    .replace( 'COMPUTE_SUB',
        'the L₂-norm of the momentum field over space at each time' )\
    .replace( 'VAR_SUB',
        'Indices of the momentum components (m₁, m₂[, m₃]) in the variable-mode of X' )\
    .replace( 'OUTPUT_SUB',
        '√(∑|momentum|²) at each time (L₂-norm)' )
)
def compute_momentum(X, var, time, exo: ExoInfo, compute_deriv=False) -> Tuple[np.floating, np.ndarray]:
    func = lambda v: np.sum(v**2,axis=2)
    deriv = lambda v: 2.0*v
    p, jac = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    if compute_deriv:
        p = np.sqrt(p)
        jac[np.ix_(range(jac.shape[0]),range(jac.shape[1]),var,time)] *= 0.5/np.reshape(p,(1,1,1,len(time)))
        return p,jac
    else:
        return np.sqrt(p), np.empty(())



@set_docstring(
    DOCSTRING_TEMPLATE_COMPUTE\
    .replace( 'COMPUTE_SUB',
        'the spatial integral of internal energy (density × temperature) at each time' )\
    .replace( 'VAR_SUB',
        'Indices of the density and temperature components (ρ, T) in the variable-mode of X' )\
    .replace( 'OUTPUT_SUB',
        'Spatial integral of ρ·T at each time' )
)
def compute_internal_energy(X, var, time, exo: ExoInfo, compute_deriv=False) -> Tuple[np.floating, np.ndarray]:
    func = lambda v: np.prod(v,axis=2)
    deriv = lambda v: v[:,:,[1,0],:]
    p, jac = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    if compute_deriv:
        return (p, jac)
    else:
        return p, np.empty(())


@set_docstring(
    DOCSTRING_TEMPLATE_COMPUTE\
    .replace( 'COMPUTE_SUB',
        'the spatial integral of magnetic energy ½·∑|B|² at each time' )\
    .replace( 'VAR_SUB',
        'Indices of the magnetic field components (B₁, B₂[, B₃]) in the variable-mode of X' )\
    .replace( 'OUTPUT_SUB',
        'Spatial integral of ½·∑|B|² at each time' )
)
def compute_magnetic_energy(X, var, time, exo: ExoInfo, compute_deriv=False) -> Tuple[np.floating, np.ndarray]:
    func = lambda v: 0.5*np.sum(v**2,axis=2)
    deriv = lambda v: v
    out = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    if compute_deriv:
        return out[0], out[1]
    else:
        return out[0], np.empty(())



@set_docstring(
    DOCSTRING_TEMPLATE_COMPUTE\
    .replace( 'COMPUTE_SUB',
       'the spatial integral of kinetic energy ½·|m|²/ρ at each time')\
    .replace( 'VAR_SUB',
        'Indices of the density and momentum components (ρ, m₁, m₂[, m₃])')\
    .replace( 'OUTPUT_SUB',
       'Spatial integral of kinetic energy at each time.')
)
def compute_kinetic_energy(X, var, time, exo: ExoInfo, compute_deriv=False) -> Tuple[np.floating, np.ndarray]:
    func = lambda v: 0.5*np.sum(v[:,:,1:,:]**2,axis=2)/v[:,:,0,:]
    def deriv(v):
        J = np.zeros(v.shape)
        J[:,:,0,:] = -0.5*np.sum(v[:,:,1:,:]**2,axis=2)/v[:,:,0,:]**2
        J[:,:,1:,:] = v[:,:,1:,:] / np.reshape(v[:,:,0,:],(J.shape[0],J.shape[1],1,J.shape[3]))
        return J
    p, jac = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    if compute_deriv:
        return (p, jac)
    else:
        return p, np.empty(())


def compute_total_energy(X, var, time, exo, compute_deriv=False) -> Tuple[np.floating, np.ndarray]:
    """
    Compute the spatial integral of total energy (internal + kinetic + magnetic).

    Parameters
    ----------
    X : ttb.ktensor or ndarray of shape (nx, ny[, nz], nvar, ntime)
        Full or reduced data tensor.  Must be
        • 4-way: (nx, ny,   nvar=6, ntime) for a 2D simulation, or
        • 5-way: (nx, ny, nz, nvar=8, ntime) for a 3D simulation.
        The ‘variable’ mode must contain exactly:
          [B₁, B₂(, B₃), ρ, m₁, m₂(, m₃), T]
        in that order (so nvar=6 when num_space=2, nvar=8 when num_space=3).

    var : sequence of int
        Indices selecting the above variables in X's variable-mode.

    time : sequence of int
        Time-step indices at which to evaluate each integral.

    exo : ExoInfo
        Provides quadrature weights and mappings for spatial integration.

    compute_deriv : bool, default=False
        If True, also compute the Jacobian of the total-energy integral
        with respect to X.

    Returns
    -------
    E : ndarray of length len(time)
        Total energy integral (E = internal + kinetic + magnetic) at each time.

    J : ndarray or empty
        If compute_deriv is True, the Jacobian ∂E/∂X of shape
        (nx, ny[, nz], nvar, ntime); otherwise an empty array.
    """
    num_space = X.ndims - 2
    if num_space == 2:
        B_var = var[0:2]
        rho_var = [var[2]]
        mom_var = var[3:5]
        T_var = [var[5]]
    elif num_space == 3:
        B_var = var[0:3]
        rho_var = [var[3]]
        mom_var = var[4:7]
        T_var = [var[7]]
    else:
        raise ValueError()

    if compute_deriv:
        T, J_T = compute_internal_energy(X,    rho_var + T_var, time, exo, compute_deriv=True)
        P, J_P = compute_kinetic_energy( X,  rho_var + mom_var, time, exo, compute_deriv=True)
        B, J_B = compute_magnetic_energy(X,              B_var, time, exo, compute_deriv=True)
        E = T + P + B
        J = J_T + J_P + J_B
        return E, J
    else:
        T, _ = compute_internal_energy(X,    rho_var + T_var, time, exo, compute_deriv=False)
        P, _ = compute_kinetic_energy( X,  rho_var + mom_var, time, exo, compute_deriv=False)
        B, _ = compute_magnetic_energy(X,              B_var, time, exo, compute_deriv=False)
        return T + P + B, np.empty(())


def compute_energies(X,rho_var,T_var,mom_var,B_var,time,exo):
    T, _ = compute_internal_energy(X, rho_var+T_var, time, exo)
    P, _ = compute_kinetic_energy(X, rho_var+mom_var, time, exo)
    B, _ = compute_magnetic_energy(X, B_var, time, exo)
    E = T + P + B
    return E,T,P,B


def plot_momentum(X,u,var,time_ind,time_val,exo,scaler):
    if time_ind is None:
        time_ind = range(X.shape[3])
    U = scaler.unscale_tensor(u.full())
    mom_X = compute_momentum(X,var,time_ind,exo)
    mom_U = compute_momentum(U,var,time_ind,exo)
    t = time_val[time_ind]

    plt.rc('font',weight='bold')
    plt.rc('axes',labelweight='bold',linewidth=1.5)
    plt.rc('lines', linewidth=1.5)
    plt.rc('xtick.major',width=1.5)
    plt.rc('xtick.minor',width=1.5)
    plt.rc('ytick.major',width=1.5)
    plt.rc('ytick.minor',width=1.5)

    fig, ax = plt.subplots()
    ax.semilogy(t,mom_X,'b-',label='Original data')
    ax.semilogy(t,mom_U,'r-',label='Reduced data')
    ax.set_xlabel("Time")
    ax.set_ylabel("L2-norm-squared of momentum")
    ax.legend(loc='lower right')

    return fig,ax


def plot_energies(X,u,rho_var,T_var,mom_var,B_var,time_ind,time_val,exo,scaler):
    if time_ind is None:
        time_ind = range(X.shape[3])
    U = scaler.unscale_tensor(u.full())
    E_X,T_X,P_X,B_X = compute_energies(X, rho_var, T_var, mom_var, B_var, time_ind, exo)
    E_U,T_U,P_U,B_U = compute_energies(U, rho_var, T_var, mom_var, B_var, time_ind, exo)
    t = time_val[time_ind]

    plt.rc('font',weight='bold')
    plt.rc('axes',labelweight='bold',linewidth=1.5)
    plt.rc('lines', linewidth=1.5)
    plt.rc('xtick.major',width=1.5)
    plt.rc('xtick.minor',width=1.5)
    plt.rc('ytick.major',width=1.5)
    plt.rc('ytick.minor',width=1.5)

    fig, axs = plt.subplots(2,2)
    #fig.set_size_inches(8,5)
    axs[0,0].plot(t, E_X, 'b-', label='Original data')
    axs[0,0].plot(t, E_U, 'r-', label='Reduced data')
    axs[0,0].set_xlabel('Time')
    axs[0,0].set_ylabel('Total Energy')
    axs[0,0].legend(loc='upper right')
    
    axs[0,1].plot(t, T_X, 'b-', label='Original data')
    axs[0,1].plot(t, T_U, 'r-', label='Reduced data')
    axs[0,1].set_xlabel('Time')
    axs[0,1].set_ylabel('Internal Energy')

    axs[1,0].plot(t, P_X, 'b-', label='Original data')
    axs[1,0].plot(t, P_U, 'r-', label='Reduced data')
    axs[1,0].set_xlabel('Time',)
    axs[1,0].set_ylabel('Kinetic Energy')
  
    axs[1,1].plot(t, B_X, 'b-', label='Original data')
    axs[1,1].plot(t, B_U, 'r-', label='Reduced data')
    axs[1,1].set_xlabel('Time')
    axs[1,1].set_ylabel('Magnetic Energy')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    return fig,axs
    

class MomentumGoal(PhysicsGoal):
    def __init__(self, X, var, time, exo):
        self.exo = exo
        super().__init__(X, var, time)

    def computeTarget(self,U,compute_deriv=False):
        return compute_momentum(U,self.var,self.time,self.exo,compute_deriv)


class InternalEnergyGoal(PhysicsGoal):
    def __init__(self, X, var, time, exo):
        self.exo = exo
        super().__init__(X, var, time)

    def computeTarget(self,U,compute_deriv=False):
        return compute_internal_energy(U,self.var,self.time,self.exo,compute_deriv)


class MagneticEnergyGoal(PhysicsGoal):
    def __init__(self, X, var, time, exo):
        self.exo = exo
        super().__init__(X, var, time)

    def computeTarget(self,U,compute_deriv=False):
        return compute_magnetic_energy(U,self.var,self.time,self.exo,compute_deriv)


class KineticEnergyGoal(PhysicsGoal):
    def __init__(self, X, var, time, exo):
        self.exo = exo
        super().__init__(X, var, time)

    def computeTarget(self,U,compute_deriv=False):
        return compute_kinetic_energy(U,self.var,self.time,self.exo,compute_deriv)


class TotalEnergyGoal(PhysicsGoal):
    def __init__(self, X, var, time, exo):
        self.exo = exo
        super().__init__(X, var, time)

    def computeTarget(self,U,compute_deriv=False):
        return compute_total_energy(U,self.var,self.time,self.exo,compute_deriv)
