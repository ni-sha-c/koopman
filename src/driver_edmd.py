import sys
from pylab import *
from numpy import *
from time import clock
from numba import jit
from matplotlib import *
from fourierAnalysis import *
sys.path.insert(0,'../examples/')
from linearcat import *

foa = FourierAnalysis()
solver = Solver()
n_samples = 5
n_dim = solver.state_dim
n_dict = 10
u = foa.solve_primal(solver, \
        solver.u_init, \
        n_samples,\
        solver.s0)

X = u[:-1]
Y = u[1:]
Psi_X = foa.compute_bases(X,n_dict)
Psi_Y = foa.compute_bases(Y,n_dict)
G = zeros((n_dict,n_dict))
A = zeros((n_dict,n_dict))
for i in range(n_samples-1):
    G += dot(Psi_X[i].reshape(n_dict,1),\
            Psi_X[i].reshape(1,n_dict))
    A +=  dot(Psi_X[i].reshape(n_dict,1),\
            Psi_Y[i].reshape(1,n_dict))

G /= 1./(n_samples-1)
A /= 1./(n_samples-1)
K = inv(G)*A
