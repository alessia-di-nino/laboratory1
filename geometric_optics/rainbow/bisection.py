import numpy as np

def f1(n):
    return (4 * np.arcsin (np.sqrt((4 - n**2) / (8*n**2))) - 2* np.arcsin (np.sqrt(4 - n**2)/(8)))/((4* np.arcsin(np.sqrt(4 - n**2)/(3*n**2))) - 2 * np.arcsin(np.sqrt(4 - n**2)/(3))) - ((495.609)/(407.901))

def bisection(f, a, b, tol=1e-14, maxiter=10000):
    if f(a)*f(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for i in range(maxiter):
        c = (a + b)/2
    
        if f(c) == 0 or (b - a)/2 < tol:
            return c
    
        if f(a)*f(c) < 0:
            b = c

        else:
            a = c
    raise RuntimeError("bisection method failed")

print(bisection(f1, 1.3, 1.4))
