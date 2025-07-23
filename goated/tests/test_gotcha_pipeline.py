import os
from pathlib import Path

import pyttb as ttb
import numpy as np

import goated.utils.exo as ex
import goated.utils.scalers as sc
import goated.goals.physics as pgoals
import goated.examples as goex

import goated.rol_interface.models as rolm
from goated.tucker import TuckerObjective, GotchaObjective, TuckerGoals


def build_tensor_goalobjects_and_scaler():
    # Read tensor from exodus file
    fname = Path(os.path.dirname(goex.__file__)) / Path('small.exo')
    exo = ex.ExoInfo()
    exo.read_sheet(fname)

    vars = [0,1,3,4,6,7,9]       # Exclude BZ, R, RHO_UZ, and U
    mag_var_ind = [0, 1]         # variable indices for magnetic field
    rho_var_ind = [2]            # variable index for density
    mom_var_ind = [4 ,5]         # variables for momentum goal
    tmp_var_ind = [6]            # variable index for temperature
    int_var_ind = rho_var_ind + tmp_var_ind
    kin_var_ind = rho_var_ind + mom_var_ind
    num_timesteps = len(exo.t)
    mom_time_ind = range(1, num_timesteps) # time steps for momenum goal
    nrg_time_ind = range(0, num_timesteps) # time steps for energy goals
    X = ttb.tensor(exo.tensor_data[:,:,vars,:])
    
    M_goal = pgoals.MomentumGoal(       X,  mom_var_ind,  mom_time_ind,  exo )
    P_goal = pgoals.KineticEnergyGoal(  X,  kin_var_ind,  nrg_time_ind,  exo )
    T_goal = pgoals.InternalEnergyGoal( X,  int_var_ind,  nrg_time_ind,  exo )
    B_goal = pgoals.MagneticEnergyGoal( X,  mag_var_ind,  nrg_time_ind,  exo )
    # scale tensor values so that each variable has roughly the same order of magnitude
    scaler = sc.StdScaler(X, exo.var_mode)
    return X, M_goal, T_goal, P_goal, B_goal, scaler

X, mom_goal, T_goal, P_goal, B_goal, scaler = build_tensor_goalobjects_and_scaler()
goals = TuckerGoals(scaler, [mom_goal, T_goal, P_goal, B_goal])
Xs = scaler.scale_tensor(X)
hosvd_tol = 0.01    # HOSVD tolerance for initial guess
us0 = ttb.hosvd(Xs, hosvd_tol, verbosity=1)
U0 = scaler.unscale_tensor(us0.full())
fit0 = 1-(X-U0).norm()/X.norm()
print(f'Core size : {np.prod(us0.core.shape)}')

mom_goal0, T_goal0, P_goal0, B_goal0 = goals.eval_goals(U0)
print(f'Initial fit = {fit0:.4f},\nmomentum goal = {mom_goal0:.2e},\ninternal energy goal = {T_goal0:.2e},\nkinetic energy goal {P_goal0:.2e},\nmagnetic energy goal = {B_goal0:.2e}')

goals = TuckerGoals(scaler, [mom_goal, T_goal, P_goal, B_goal])
ng = len(goals.goals)+1
a = 1/(ng*((Xs-us0.full()).norm()**2))
b = 1.0
goals.auto_reweight(U0)
gotcha = GotchaObjective(Xs, scaler, goals, a, b)

problem = rolm.GoatedRolModel(gotcha, us0)
problem.solve()

us1 = problem.decomp
U1 = scaler.unscale_tensor(us1.full())
fit1 = 1-(X-U1).norm()/X.norm()
mom_goal1, T_goal1, P_goal1, B_goal1 = goals.eval_goals(U1)
print(f'Final fit = {fit1:.4f},\nmomentum goal = {mom_goal1:.2e},\ninternal energy goal = {T_goal1:.2e},\nkinetic energy goal {P_goal1:.2e},\nmagnetic energy goal = {B_goal1:.2e}')



print()
