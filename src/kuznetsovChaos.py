import sys
from pylab import *
from numpy import *
from time import clock
from numba import jit
from matplotlib import *
from mpl_toolkits.mplot3d import Axes3D
from fourierAnalysis import *
from scipy import interpolate
sys.path.insert(0,'../examples/')
from arnoldcat import *

@jit(nopython=True) 
def objective(u):
    return u[0]*u[1]

@jit(nopython=True)
def grad_objective(u):
    gradJ = zeros_like(u)
    gradJ[0] = u[1]
    gradJ[1] = u[0]
    return gradJ



def plot_timeshifted_function(u,f):
    x,y = u[0],u[1]
    ntimes = f.shape[0]
    xx = linspace(min(x), max(x), 500)
    yy = linspace(min(y), max(y), 500)
    xx, yy = meshgrid(xx,yy)
    f_0 = interpolate.griddata((x,y),f[0],(xx,yy),method='nearest')
    f_mid = interpolate.griddata((x,y),f[ntimes//2],(xx,yy),method='nearest')
    f_end = interpolate.griddata((x,y),f[ntimes-1],(xx,yy),method='nearest')
    
    fig, ax = subplots(1,3,figsize=(20,15)) 
    ax[0].tick_params(labelsize=30)
    ax[1].tick_params(labelsize=30)
    ax[2].tick_params(labelsize=30)
    time_0 = ax[0].contourf(xx,yy,f_0)
    ax[0].set_xlabel("t = 0",fontsize=30)
    time_mid = ax[1].contourf(xx,yy,f_mid)
    ax[1].set_xlabel("t = 3",fontsize=30)
    time_end = ax[2].contourf(xx,yy,f_end)
    ax[2].set_xlabel("t = 20",fontsize=30)
    fig.colorbar(time_end)
    savefig("../examples/plots/presentation/timeshifted_objective.png", dpi=500)

@jit(nopython=True)
def solve_primal(solver_map, u_init, n_steps, s):
    u = empty((n_steps, u_init.size))
    u[-1] = u_init
    for i in range(n_steps):
        u[i] = solver_map.primal_step(u[i-1],s,1)
    return u

    
def solve_unstable_direction(self, solver_map,\
            u, v_init, n_steps, s):
    v = empty((n_steps, v_init.size))
    v[0] = v_init
    v[0] /= linalg.norm(v[0])
    for i in range(1,n_steps):
        v[i] = dot(solver_map.gradFs(u[i-1],s),v[i-1])
        v[i] /= linalg.norm(v[i])
    return v

foa = FourierAnalysis()
solver = Solver()
n_runup_steps = 10000
n_samples = 100
n_steps = 100
u = foa.solve_primal(solver, \
        solver.u_init, \
        n_runup_steps,\
        solver.s0)
u_0 = u[-n_samples:]
u_nsteps =  foa.solve_primal(solver, \
        u_0[0], \
        n_steps + n_samples,\
        solver.s0)
u_nsteps = u_nsteps[n_steps::n_steps]

v_init = rand(2)
v = foa.solve_unstable_direction( \
        solver, u, v_init, n_samples, \
        solver.s0)

v = v[-n_samples:]

#J_trj = foa.compute_timeshifted_objective( \
#        solver, objective, u, 5).T
#plot_timeshifted_function(u.T,J_trj)
dJ_trj = foa.compute_directional_derivative_timeshifted_objective(\
        solver,grad_objective,u,v,5).T
plot_timeshifted_function(u.T,dJ_trj)
