import numpy as np
import matplotlib.pyplot as plt
from goated.goals.abstract import Goal, CPGoal


def compute_momentum(X, var, time, exo, compute_deriv=False):
    func = lambda v: np.sum(v**2,axis=2)
    deriv = lambda v: 2.0*v
    out = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    if compute_deriv:
        p = out[0]
        jac = out[1]
        p = np.sqrt(p)
        jac[np.ix_(range(jac.shape[0]),range(jac.shape[1]),var,time)] *= 0.5/np.reshape(p,(1,1,1,len(time)))
        return p,jac
    else:
        return np.sqrt(out)


def compute_internal_energy(X, var, time, exo, compute_deriv=False):
    func = lambda v: np.prod(v,axis=2)
    deriv = lambda v: v[:,:,[1,0],:]
    out = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    return out


def compute_magnetic_energy(X, var, time, exo, compute_deriv=False):
    func = lambda v: 0.5*np.sum(v**2,axis=2)
    deriv = lambda v: v
    out = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    return out


def compute_kinetic_energy(X, var, time, exo, compute_deriv=False):
    func = lambda v: 0.5*np.sum(v[:,:,1:,:]**2,axis=2)/v[:,:,0,:]
    def deriv(v):
        J = np.zeros(v.shape)
        J[:,:,0,:] = -0.5*np.sum(v[:,:,1:,:]**2,axis=2)/v[:,:,0,:]**2
        J[:,:,1:,:] = v[:,:,1:,:] / np.reshape(v[:,:,0,:],(J.shape[0],J.shape[1],1,J.shape[3]))
        return J
    out = exo.compute_spatial_integral(X, var, time, func=func, deriv=deriv, compute_func=True, compute_deriv=compute_deriv)
    return out


def compute_total_energy(X, var, time, exo, compute_deriv=False):
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

    if compute_deriv:
        T,J_T = compute_internal_energy(X, rho_var+T_var, time, exo)
        P,J_P = compute_kinetic_energy(X, rho_var+mom_var, time, exo)
        B,J_B = compute_magnetic_energy(X, B_var, time, exo)
        E = T + P + B
        J = J_T + J_P + J_B
        return E,J
    else:
        T = compute_internal_energy(X, rho_var+T_var, time, exo)
        P = compute_kinetic_energy(X, rho_var+mom_var, time, exo)
        B = compute_magnetic_energy(X, B_var, time, exo)
        return T + P + B


def compute_energies(X,rho_var,T_var,mom_var,B_var,time,exo):
    T = compute_internal_energy(X, rho_var+T_var, time, exo)
    P = compute_kinetic_energy(X, rho_var+mom_var, time, exo)
    B = compute_magnetic_energy(X, B_var, time, exo)
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
    

class MomentumGoal(Goal):
    def __init__(self, X, var, time, exo):
        super().__init__(X,var,time,exo)

    def computeTarget(self,X,compute_deriv=False):
        return compute_momentum(X,self.var,self.time,self.exo,compute_deriv=compute_deriv)


class InternalEnergyGoal(Goal):
    def __init__(self, X, var, time, exo):
        super().__init__(X,var,time,exo)

    def computeTarget(self,X,compute_deriv=False):
        return compute_internal_energy(X,self.var,self.time,self.exo,compute_deriv=compute_deriv)


class MagneticEnergyGoal(Goal):
    def __init__(self, X, var, time, exo):
        super().__init__(X,var,time,exo)

    def computeTarget(self,X,compute_deriv=False):
        return compute_magnetic_energy(X,self.var,self.time,self.exo,compute_deriv=compute_deriv)


class KineticEnergyGoal(Goal):
    def __init__(self, X, var, time, exo):
        super().__init__(X,var,time,exo)

    def computeTarget(self,X,compute_deriv=False):
        return compute_kinetic_energy(X,self.var,self.time,self.exo,compute_deriv=compute_deriv)


class TotalEnergyGoal(Goal):
    def __init__(self, X, var, time, exo):
        super().__init__(X,var,time,exo)

    def computeTarget(self,X,compute_deriv=False):
        return compute_total_energy(X,self.var,self.time,self.exo,compute_deriv=compute_deriv)
