import sys
from pylab import *
from numpy import *
from time import clock
from numba import jit
from matplotlib import *
from fourierAnalysis import *
from utils import *
sys.path.insert(0,'../examples/')
from arnoldcat import *

foa = FourierAnalysis()
solver = Solver()
helper_funs = HelperFunctions()
n_samples = 1000
n_dim = solver.state_dim
n_poly_order = 2
n_basis = n_poly_order**n_dim
is_attractor = True
if(is_attractor):
    u = foa.solve_primal(solver, \
        solver.u_init, \
        n_samples,\
        solver.s0)
    X = u[:-1]
    Y = u[1:]
    Psi_u = helper_funs.compute_legendre_bases(u,n_poly_order)
    Psi_X = Psi_u[:-1]
    Psi_Y = Psi_u[1:]

else:
    X = rand(n_samples, \
            solver.u_init.size)
    Y = foa.solve_onestep(solver,\
            X, solver.s0)
    Psi_X = helper_funs.compute_legendre_bases(X,n_poly_order)
    Psi_Y = helper_funs.compute_legendre_bases(Y,n_poly_order)
    

G = zeros((n_basis,n_basis))
A = zeros((n_basis,n_basis))
for i in range(n_samples-1):
    G += dot(Psi_X[i].reshape(n_basis,1),\
            Psi_X[i].reshape(1,n_basis))
    A +=  dot(Psi_X[i].reshape(n_basis,1),\
            Psi_Y[i].reshape(1,n_basis))

G /= 1./(n_samples-1)
A /= 1./(n_samples-1)
K = dot(inv(G),A)
K_eig_values, K_eig_vecs = eig(K)
