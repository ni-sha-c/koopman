from numpy import *
import sys
sys.path.insert(0,'../examples/')
from arnoldcat import *
from numba import jit

@jit(nopython=True)
def solve_primal(solver, u0, s0, n=1):
    u_trj = zeros((n, u0.size))
    u_trj[-1] = u0
    for i in range(n):
        u_trj[i] = solver.primal_step(\
                u_trj[i-1], s0, 1)
    return u_trj



solver = Solver()
s0 = solver.s0
n_trj = 10000
u_init = solver.u_init
u_init = solver.primal_step(u_init, s0, \
        100)
u_trj = solve_primal(solver, u_init, s0, \
        n_trj)

