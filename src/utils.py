from numpy import *
from numba import jit

@jit(nopython=True)
def fact(x):
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

@jit(nopython=True)
def n_choose_k(n,k):
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
    return res/fact(k)

@jit(nopython=True)
def legendre_fun_nd(x,n):
    (n_points, n_dim) = x.shape
    pn_x_comp = zeros(x.shape)
    for k in range(n+1):
        pn_x_comp += n_choose_k(n,k)**2.0*\
                (0.5*(x-1.))**(n-k)*\
                (0.5*(x+1.))**k
    pn_x = ones(n_points)
    for n in range(n_points):
        pn_x[n] = prod(pn_x_comp[n])
    return pn_x


@jit(nopython=True)
def legendre_fun_1d(x,n):
    pn_x = zeros(x.shape)
    for k in range(n+1):
        pn_x += n_choose_k(n,k)**2.0*\
                (0.5*(x-1.))**(n-k)*\
                (0.5*(x+1.))**k
    return pn_x

