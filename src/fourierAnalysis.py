#Statistical sensitivity analysis algorithm for maps.
from pylab import *
from numpy import *
from numba import jitclass
from numba import float64, int64
spec = []

@jitclass(spec)
class FourierAnalysis:
    def __init__(self):
        pass

    def solve_primal(self, solver_map, u_init, n_steps, s):
        u = empty((n_steps, u_init.size))
        u[0] = u_init
        for i in range(1,n_steps):
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
    
    
    def solve_unstable_adjoint_direction(self, solver_map,\
            u, w_init, n_steps, s):
        w = empty((n_steps, w_init.size))
        w[-1] = w_init
        w[-1] /= norm(w[-1])
        for i in range(n_steps-1,0,-1):
            w[i-1] = solver_map.adjoint_step(w[i],u[i-1],s,0)
            w[i-1] /= norm(w[i-1])
        return w
    
    
    
    def compute_source_tangent(self, solver_map, \
            u, n_steps, s0):
        param_dim = s0.size
        dFds = zeros((n_steps,param_dim,u.shape[1]))
        for i in range(n_steps):
            dFds[i] = solver_map.DFDs(u[i],s0)
        return dFds




    def compute_fourier_transform_pullback(self,u,f,xi,\
            n,n_samples,decorr_len):
        g = exp(1j*dot(u,xi))
        expf_expg = mean(f)*mean(g)
        return self.compute_correlation_function\
                (f,g,n,n_samples,decorr_len) - \
                expf_expg
        


    def compute_correlation(self,f,g,n,n_samples,decorr_len):
        corr = 0.0
        N = f.shape[0]-n
       
        fbar = mean(f)
        gbar = mean(g)
        for i in range(0,N,decorr_len):
            corr += (f[i] - fbar)*(g[n+i] - gbar)/n_samples
        return corr

    def compute_correlation_function(self,f,g,n_max,\
            n_samples,decorr_len):
        corr_fg = zeros(n_max)
        for n in range(n_max):
            corr_fg[n] = self.compute_correlation(f,g,n,\
                    n_samples,decorr_len)
        return corr_fg

            
    def compute_gaussian_bases(self,x,n_funs=10):
        n_points = x.shape[0]
        n_dim = x.shape[1]
        f = zeros((n_points, n_funs))
        mu = zeros((n_funs, n_dim))
        inv_sigma = eye(n_dim)
        for i in range(n_dim):
            mu_min = 0.
            mu_max = 1.
            mu[:,i] = linspace(mu_min,\
                    mu_max, n_funs)
        for i in range(n_points):
            for j in range(n_funs):
                f[i,j] = exp(-0.5*(dot(x[i]-mu[j],\
                        dot(inv_sigma,x[i]-mu[j]))))
                f[i,j] /= sqrt(2.0*pi*det(inv_sigma))
        return f


    def compute_hermite_bases(self,x,n_funcs=10):
        n_points = x.shape[0]
        n_dim = x.shape[1]
        f = zeros((n_points, n_funs))
        mu = zeros((n_funs, n_dim))

        for n in range(n_funs):
            for i in range(n_points):
                f[i,n] = polynomial.hermite.hermval(x[i],
        return f

