from numpy import *
from numba import jitclass
from numba import float64, int64
spec = []

@jitclass(spec)
class HelperFunctions:
    def __init__(self):
        pass
    def fact(self,x):
        if(int(x) != x):
            return -1
        if(x < 0):
            return -1
        if(x==0 or x==1):
            return 1
        x_fact = 1
        for n in range(1,x+1):
            x_fact *= n
        return x_fact

    def n_choose_k(self,n,k):
        if(int(n) != n):
            return -1
        if(int(k) != k):
            return -1 
        if(n < k):
            return -1
        if(k < 0):
            return -1
        if(k==0):
            return 1
        res = 1
        i = 0
        j = n
        while(i < k):
            res *= j
            j -= 1
            i += 1
        return res/self.fact(k)

    def legendre_fun_nd(self,x,n):
        #n is a multiindex of degrees
        (n_points, n_dim) = x.shape
        x = x.T
        pn_x_comp = zeros(x.shape)
        for i,n_i in enumerate(n):
            pn_x_comp[i] = self.legendre_fun_1d(x[i],n_i)
        return pn_x_comp


    def legendre_fun_1d(self,x,n):
        pn_x = zeros(x.shape)
        for k in range(n+1):
            pn_x += self.n_choose_k(n,k)**2.0*\
                    (0.5*(x-1.))**(n-k)*\
                    (0.5*(x+1.))**k
        return pn_x



    def compute_legendre_bases(self,x,n_funs=5):
        n_points = x.shape[0]
        n_dim = x.shape[1]
        f = zeros((n_funs, n_dims, n_points))
        for n in range(n_funs):
            f[n] = self.legendre_fun_nd(x, n)
        return f.T

    def tensor_product(self,A):
        d = A.shape[0]
        n = A.shape[1]
        A_old = A[0].reshape((1,n))
        for i in range(1,d):
            A_new = empty((n**i,n))
            for j in range(n**(i-1)):
                A_new[j*n:(j+1)*n] = dot(\
                    A_old[j].copy().reshape((n,1)),\
                    A[i].reshape((1,n)))
            A_old = A_new.copy()
        return A_new.ravel()            
