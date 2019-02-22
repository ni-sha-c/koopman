from numpy import *
import sys
from numba import jit
from matplotlib.pyplot import *
sys.path.insert(0,'../examples/')
from arnoldcat import *


#@jit(nopython=True)
def compute_eigenfunction(v, x):
    n_nodes = v.shape[0]
    x_grid = linspace(0, 1.,n_nodes+1)
    delta_x = x_grid[1] - x_grid[0]
    v_x = empty_like(x, dtype="complex")
    n_points = x.shape[1]
    n_dim = x.shape[0]
    nodes_x = zeros((n_dim, n_points), dtype="int")
    nodes = zeros(n_points, dtype="int")
    n_nodes_per_dim = int(n_nodes**(1/n_dim))
    for k in range(n_dim):
        nodes_x[k] = build_chain(x[k],n_nodes_per_dim) 
        nodes += nodes_x[k]*(n_nodes_per_dim**k)
    v_x = v[nodes]
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
def solve_multiple_init_cond(solver, u0, s0, n=1):
    n_trj = u0.shape[0]
    n_dim = u0.shape[1]
    u_trj = zeros((n_trj, n_dim))
    for i in range(n_trj):
        u_trj[i] = solver.primal_step(\
                u0[i], s0, n)
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


@jit(nopython=True)
def build_transition_matrix(chain, n_nodes):
    P = zeros((n_nodes,n_nodes))
    chain_length = chain.shape[0]
    for i in range(1,chain_length):
        P[chain[i], chain[i-1]] += 1
    for i in range(n_nodes):
        p = sum(P[i])
        if(p >= 1):
            P[i] /= p
    return P

#@jit(nopython=True)
def build_chain(x, n_nodes):
    x_grid = linspace(0, 1.,n_nodes+1)
    delta_x = x_grid[1] - x_grid[0]
    n_points = x.shape[0]
    chain_x = empty(n_points, dtype="int")
    for j,xj in enumerate(x):
        for i in range(n_nodes + 1):
            if(abs(xj - x_grid[i]) < delta_x):
                node_x = i
                break
        chain_x[j] = node_x
    return chain_x

@jit(nopython=True)
def analytical_transition_matrix(P):
    n_nodes = P.shape[0]
    for i in range(n_nodes):
        for j in range(n_nodes):
            if(P[i,j] > 0.2):
                P[i,j] = 0.25
    return P


solver = Solver()
s0 = solver.s0
s0[0] = 0.01
n_trj = 10000
u_init = solver.u_init
u_init = solver.primal_step(u_init, s0, \
        100)
u_trj = solve_primal(solver, u_init, s0, \
        n_trj)
u_trj = u_trj.T
n_dim = solver.state_dim
n_nodes_per_dim = 27
chain_1d = empty((n_dim, n_trj), dtype='int')
chain = zeros(n_trj, dtype='int')
for i in range(n_dim):
    chain_1d[i] = build_chain(u_trj[i], n_nodes_per_dim)
    chain += chain_1d[i]*(n_nodes_per_dim**i)    
n_nodes = n_nodes_per_dim**n_dim
P = build_transition_matrix(chain, n_nodes)
P = analytical_transition_matrix(P)
l, V = eig(P.T)


n_grid = 100
x_grid = linspace(0.,1.-1.e-3,n_grid)
x_grid, y_grid = meshgrid(x_grid,x_grid)
eig_index = 1
vi = V[:,eig_index]
f_grid = empty_like(x_grid,dtype="complex")
Kf_grid = empty_like(x_grid,dtype="complex")
Pf_grid = empty_like(x_grid,dtype="complex")
u_grid = array([hstack(x_grid),hstack(y_grid)])
f_grid = compute_eigenfunction(vi,u_grid)
Kx_grid = solve_multiple_init_cond(solver, u_grid.T, s0, 1)
Kf_grid = compute_eigenfunction(vi, Kx_grid.T)

f_grid = reshape(f_grid, (n_grid, n_grid))
Kf_grid = reshape(Kf_grid, (n_grid, n_grid))

fig = figure(figsize=[15,10])
ax = fig.add_subplot(121) 
ax1 = fig.add_subplot(122)
im = ax.contourf(x_grid, y_grid, \
        real(l[eig_index]*f_grid), label="$\Phi$")
ax1.contourf(x_grid, y_grid,\
        real(Kf_grid), linewidth=3.0,label="$K\Phi$")
ax.set_xlabel("x",fontsize=24)
ax.set_title("Real part of $\lambda \Psi$",fontsize=24)
ax.tick_params(axis="both",labelsize=24)
ax1.set_xlabel("x",fontsize=24)
ax1.set_title("Real part of $K\Psi$",fontsize=24)
ax1.tick_params(axis="both",labelsize=24)
cax = fig.add_axes([0.9, 0.1, 0.01, 0.7])
cb = colorbar(im, ax=[ax,ax1], cax=cax)
cb.ax.yaxis.set_tick_params(labelsize=24)
tight_layout(pad=2,rect=[0.01,0.01,0.9,0.9])
savefig("../examples/plots/arnoldUlam_lefteigenvectors_P_real.png")

fig = figure(figsize=[15,10])
ax = fig.add_subplot(121) 
ax1 = fig.add_subplot(122)
ax.contourf(x_grid, y_grid, \
        imag(l[eig_index]*f_grid))
ax1.contourf(x_grid, y_grid,\
        imag(Kf_grid))
ax.set_xlabel("x",fontsize=24)
ax.set_title("Imag part of $\lambda \Psi$",fontsize=24)
ax.tick_params(axis="both",labelsize=24)
ax1.set_xlabel("x",fontsize=24)
ax1.set_title("Imag part of $K\Psi$",fontsize=24)
ax1.tick_params(axis="both",labelsize=24)
cax = fig.add_axes([0.9, 0.1, 0.01, 0.7])
cb = colorbar(im, ax=[ax,ax1], cax=cax)
cb.ax.yaxis.set_tick_params(labelsize=24)
tight_layout(pad=2,rect=[0.01,0.01,0.9,0.9])
savefig("../examples/plots/arnoldUlam_lefteigenvectors_P_imag.png")


fig = figure(figsize=[15,10])
ax = fig.add_subplot(111) 
ax.plot(real(l), imag(l), "k.", ms = 15)
ax.set_xlabel("Real part of eigenvalues of P",fontsize=20)
ax.set_ylabel("Im part of eigenvalues of P",fontsize=20)
ax.set_title("Frobenius-Perron spectra of the Arnold's Cat Map",fontsize=20)
ax.tick_params(axis="both",labelsize=24)
A = array([2.,1.,1.,1.]).reshape(2,2)
lA, VA = eig(A)
r = min(lA)
x = linspace(-r,r,50)
y = sqrt(r*r - x*x)
ax.plot(x, y, "k-.", linewidth=3.0,alpha=0.5)
ax.plot(x, -y, "k-.", linewidth=3.0,alpha=0.5)
tight_layout(pad=2,rect=[0.01,0.01,0.9,0.9])
savefig("../examples/plots/arnoldUlam_eigenvalues.png")

fig = figure(figsize=[15,10])
ax = fig.add_subplot(121) 
ax1 = fig.add_subplot(122)
ax.contourf(x_grid, y_grid, \
        real(l[eig_index]*f_grid))
Kx_grid = solve_multiple_init_cond(solver, u_grid.T, s0, 3)
Kf_grid = compute_eigenfunction(vi, Kx_grid.T)
Kf_grid = reshape(Kf_grid, (n_grid, n_grid))
ax1.contourf(x_grid, y_grid,\
        real(Kf_grid))
n_grid = 5
eps = 1.e-1
v_stable = VA[:,1]
v_unstable = VA[:,0]
x_grid = linspace(0.2,0.8,n_grid)
x_grid, y_grid = meshgrid(x_grid,x_grid)
x_grid = hstack(x_grid)
y_grid = hstack(y_grid)
x_grid_plus_vA = x_grid + eps*v_stable[0]
y_grid_plus_vA = y_grid + eps*v_stable[1]
ax1.plot([x_grid,x_grid_plus_vA],\
        [y_grid,y_grid_plus_vA],\
        linewidth=3.0, color='b', label="stable")
x_grid_plus_vA = x_grid + eps*v_unstable[0]
y_grid_plus_vA = y_grid + eps*v_unstable[1]
ax1.plot([x_grid,x_grid_plus_vA],\
        [y_grid,y_grid_plus_vA],\
        linewidth=3.0, color='r', label="unstable")

ax.set_xlabel("x",fontsize=24)
ax.set_title("Real part of $\lambda \Psi$",fontsize=24)
ax.tick_params(axis="both",labelsize=24)
ax1.set_xlabel("x",fontsize=24)
ax1.set_title("Real part of $K\Psi$",fontsize=24)
ax1.tick_params(axis="both",labelsize=24)
cax = fig.add_axes([0.9, 0.1, 0.01, 0.7])
cb = colorbar(im, ax=[ax,ax1], cax=cax)
cb.ax.yaxis.set_tick_params(labelsize=24)
tight_layout(pad=2,rect=[0.01,0.01,0.9,0.9])
savefig("../examples/plots/arnoldUlam_stable_unstable_directions.png")




'''
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
