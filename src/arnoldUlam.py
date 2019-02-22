from numpy import *
import sys
from numba import jit
from matplotlib.pyplot import *

@jit(nopython=True)
def generate_ulam_matrix(n_nodes):
    P = zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        P[2*i % n_nodes, i] = 1/2.
        P[(2*i + 1) % n_nodes, i] = 1/2.
    return P

@jit(nopython=True)
def compute_eigenfunction(v, x):
    n_nodes = v.shape[0]
    x_grid = linspace(0, 1.,n_nodes+1)
    delta_x = x_grid[1] - x_grid[0]
    v_x = empty_like(x, dtype="complex")
    for j,xj in enumerate(x):
        for i in range(n_nodes + 1):
            if(abs(xj - x_grid[i]) < delta_x):
                node_x = i
                break
            v_x[j] = v[node_x]
    return v_x
    
@jit(nopython=True)
def compute_transfer_eigenfunction(v, x):
    n_nodes = v.shape[0]
    x_grid = linspace(0., 1., n_nodes + 1)
    delta_x = x_grid[1] - x_grid[0]
    for i in range(n_nodes + 1):
        if(abs(x/2 - x_grid[i]) < delta_x):
            node_x = i
            break
    node_x_1 = node_x
    node_x_2 = (node_x + n_nodes//2) % n_nodes
    return v[node_x_1]/2 + v[node_x_2]/2

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


'''
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

n_nodes = 4
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
'''
'''
n_nodes = 35
P = generate_ulam_matrix(n_nodes)
l, V = eig(P)
n_grid = 100
x_grid = linspace(0.,1.-1.e-3,n_grid)
eig_index = 5
vi = V[:,eig_index]
f_grid = empty_like(x_grid,dtype="complex")
Kf_grid = empty_like(x_grid,dtype="complex")
Pf_grid = empty_like(x_grid,dtype="complex")
for i in range(n_grid):
    f_grid[i] = compute_eigenfunction(vi,x_grid[i])
    Kf_grid[i] = compute_transfer_eigenfunction(vi,x_grid[i])

fig = figure(figsize=[15,10])
ax = fig.add_subplot(121) 
ax1 = fig.add_subplot(122)
ax.plot(x_grid, real(l[eig_index]*f_grid), linewidth=3.0,label="$\Phi$")
ax.plot(x_grid, real(Kf_grid), linewidth=3.0,label="$P\Phi$")
ax.set_xlabel("x",fontsize=24)
#ax.set_title("Are these really eigenfunctions?",fontsize=24)
ax.tick_params(axis="both",labelsize=24)
ax.legend()
ax1.plot(x_grid, imag(l[eig_index]*f_grid), linewidth=3.0,label="$\Phi$")
ax1.plot(x_grid, imag(Kf_grid), linewidth=3.0,label="$P\Phi$")
ax1.set_xlabel("x",fontsize=24)
#ax1.set_title("Are these really eigenfunctions?",fontsize=24)
ax1.tick_params(axis="both",labelsize=24)
ax1.legend(fontsize=24)
ax.legend(fontsize=24)
savefig("../examples/plots/sawtoothUlam_transfer_eigenfunction_iterate.png")


fig = figure(figsize=[15,10])
ax = fig.add_subplot(111) 
ax.plot(real(l), imag(l), "k.", ms = 15)
ax.set_xlabel("Real part of eigenvalues of P",fontsize=20)
ax.set_ylabel("Im part of eigenvalues of P",fontsize=20)
ax.set_title("Frobenius-Perron spectra of the Sawtooth Map",fontsize=24)
ax.tick_params(axis="both",labelsize=24)
r = 0.5
x = linspace(-r,r,50)
y = sqrt(r*r - x*x)
ax.plot(x, y, "k-.", alpha=0.5)
ax.plot(x, -y, "k-.", alpha=0.5)
savefig("../examples/plots/sawtoothUlam_eigenvalues.png")

fig = figure(figsize=[15,10])
ax = fig.add_subplot(121) 
ax1 = fig.add_subplot(122)
n_fac = 5
x_grid = linspace(0.,1.,n_nodes*n_fac)
n_plots = 5
for i in range(n_plots):
    Vi_grid = tile(V[:,i],(n_fac,1)).T.\
            reshape(n_nodes*n_fac,1)
    ax.plot(x_grid, real(Vi_grid), linewidth=2.0)
    ax1.plot(x_grid, imag(Vi_grid), linewidth=2.0)
ax.set_xlabel("x",fontsize=24)
ax.set_title("real part of right eigenvectors of P",fontsize=20)
ax.tick_params(axis="both",labelsize=24)
ax1.set_xlabel("x",fontsize=24)
ax1.set_title("imag part of right eigenvectors of P",fontsize=20)
ax1.tick_params(axis="both",labelsize=24)
savefig("../examples/plots/sawtoothUlam_transfer_eigenvectors.png")

'''
