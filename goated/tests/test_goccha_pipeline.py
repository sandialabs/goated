import os
from pathlib import Path

import pyttb as ttb
import numpy as np

import goated.utils.exo as ex
import goated.utils.scalers as sc
import goated.goals.physics as goals
import goated.examples as goex

import goated.rol_interface.models as rolm
from goated.cp import GocchaObjective, CPGoals


def build_tensor_goalobjects_and_scaler():
    # Read tensor from exodus file
    fname = Path(os.path.dirname(goex.__file__)) / Path('small.exo')
    exo = ex.ExoInfo()
    exo.read_sheet(fname)
    # Some global parameters
    vars = [0,1,3,4,6,7,9]       # Exclude BZ, R, RHO_UZ, and U
    mag_var_ind = [0, 1]         # variable indices for magnetic field
    rho_var_ind = [2]            # variable index for density
    mom_var_ind = [4 ,5]         # variables for momentum goal
    tmp_var_ind = [6]            # variable index for temperature
    tot_var_ind = mag_var_ind + rho_var_ind + mom_var_ind + tmp_var_ind
    int_var_ind = rho_var_ind + tmp_var_ind
    kin_var_ind = rho_var_ind + mom_var_ind
    num_timesteps = len(exo.t)
    mom_time_ind = range(1, num_timesteps) # time steps for momenum goal
    nrg_time_ind = range(0, num_timesteps) # time steps for energy goals
    # select subset of variables
    X = ttb.tensor(exo.tensor_data[:,:,vars,:])
    M_goal = goals.MomentumGoal(       X,  mom_var_ind,  mom_time_ind,  exo )
    E_goal = goals.TotalEnergyGoal(    X,  tot_var_ind,  nrg_time_ind,  exo )
    P_goal = goals.KineticEnergyGoal(  X,  kin_var_ind,  nrg_time_ind,  exo )
    T_goal = goals.InternalEnergyGoal( X,  int_var_ind,  nrg_time_ind,  exo )
    B_goal = goals.MagneticEnergyGoal( X,  mag_var_ind,  nrg_time_ind,  exo )
    # scale tensor values so that each variable has roughly the same order of magnitude
    scaler = sc.StdScaler(X, exo.var_mode)
    return X, M_goal, E_goal, T_goal, P_goal, B_goal, scaler

X, mom_goal, E_goal, T_goal, P_goal, B_goal, scaler = build_tensor_goalobjects_and_scaler()
Xs = scaler.scale_tensor(X)
np.random.seed(1997)
rank = 15
us0,_,_ = ttb.cp_als(Xs, rank=rank, maxiters=50)
U0 = scaler.unscale_tensor(us0.full())
fit0 = 1-(X-U0).norm()/X.norm()

mom_goal0 = mom_goal.computeValue(U0)
E_goal0 = E_goal.computeValue(U0)
T_goal0 = T_goal.computeValue(U0)
P_goal0 = P_goal.computeValue(U0)
B_goal0 = B_goal.computeValue(U0)

print(f'Initial fit = {fit0:.4f},\nmomentum goal = {mom_goal0:.2e},\ntotal energy goal = {E_goal0:.2e},\ninternal energy goal = {T_goal0:.2e},\nkinetic energy goal {P_goal0:.2e},\nmagnetic energy goal = {B_goal0:.2e}')

goal = CPGoals(
    scaler,
    goals=[mom_goal, T_goal, P_goal, B_goal],
    weights=[0.25/mom_goal0, 0.25/T_goal0, 0.25/P_goal0, 0.25/B_goal0 ]
)
a = 0.2 / (Xs-us0.full()).norm()**2
goccha = GocchaObjective(Xs, goal, a, 1.0)
_ = us0.normalize(weight_factor='all')

problem = rolm.GoatedRolModel(goccha, us0)
problem.solve()

us1 = problem.decomp
U1 = scaler.unscale_tensor(us1.full())
fit1 = 1-(X-U1).norm()/X.norm()
mom_goal1 = mom_goal.computeValue(U1)
E_goal1 = E_goal.computeValue(U1)
T_goal1 = T_goal.computeValue(U1)
P_goal1 = P_goal.computeValue(U1)
B_goal1 = B_goal.computeValue(U1)

print(f'Final fit = {fit1:.4f},\nmomentum goal = {mom_goal1:.2e},\ntotal energy goal = {E_goal1:.2e},\ninternal energy goal = {T_goal1:.2e},\nkinetic energy goal {P_goal1:.2e},\nmagnetic energy goal = {B_goal1:.2e}')


print()
