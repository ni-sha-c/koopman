from numpy import *
import sys
sys.path.insert(0,'../examples/')
from sawtooth import *
from numba import jit

@jit(nopython=True)
def solve_primal(solver, u0, s0, n=1):
    u_trj = zeros((n, u0.size))
    u_trj[-1] = u0
    for i in range(n):
        u_trj[i] = solver.primal_step(\
                u_trj[i-1], s0, 1)
    return u_trj

@jit(nopython=True)
def solve_tangent(solver, u_trj, v0):
    v_trj = empty_like(u_trj)
    v_trj[0] = v0
    n = u_trj.shape[0]
    tangent_step = solver.tangent_step
    for i in range(1,n):
        v_trj[i] = tangent_step(u_trj[i-1],\
                v_trj[i-1])
    return v_trj

@jit(nopython=True)
def observable(u):
    f = empty(u.shape[0])
    f = u.T[0]
    return f



solver = Solver()
s0 = solver.s0
s0[0] = 0.01
n_trj = 10000
u_init = solver.u_init
u_init = solver.primal_step(u_init, s0, \
        100)
u_trj = solve_primal(solver, u_init, s0, \
        n_trj)

f_trj = observable(u_trj)

f_min, f_max, f_sigma = min(f_trj), max(f_trj), \
        std(f_trj)/sqrt(n_trj)

n_nodes = 10
f_grid = linspace(f_min - f_sigma, \
        f_max + f_sigma, n_nodes + 1)
f_delta = f_grid[1]-f_grid[0]
chain = empty(n_trj, dtype='int')
P = zeros((n_nodes, n_nodes))
f0 = f_trj[0]

for i, fi in enumerate(f_grid):
    if(abs(f0 - fi) < f_delta):
        chain[0] = i
        break

for n in range(1,n_trj):
    fn = f_trj[n]
    for i, fi in enumerate(f_grid):
        if(abs(fn - fi) < f_delta):
            chain[n] = i
            P[chain[n],chain[n-1]] += 1.
            break
    
for i in range(n_nodes):
    p = sum(P[i])
    P[i] /= p
