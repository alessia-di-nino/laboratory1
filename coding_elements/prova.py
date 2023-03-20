import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

x = np.array([1., 2., 3., 4., 5., 6., 7.])
y = np.array([2.12, 4.02, 5.78, 8.22, 10.23, 11.88, 13.76])
dy = np.full(y.shape, 0.25)

def f(x, m):
    return m*x

def chisq(x, y, dy, m):
    return (((y - f(x, m))/dy)**2.).sum()

plt.figure("dati")
plt.errorbar(x, y, dy, fmt = ".")
plt.xlabel("x [u.a.]")
plt.ylabel("y [u.a.]")

for m in (1., 1.5, 2., 2.5, 3.):
    chi2 = chisq(x, y, dy, m)
    plt.plot(x, f(x, m), label = f"m = {m}, $\\chi^2$ = {chi2:.2f}")
    print(chisq(x, y, dy, m))
plt.legend()

plt.figure("Chi quadro")
m = np.linspace(1., 2., 100)
chi2 = np.array([chisq(x, y, dy, m_) for m_ in m])
plt.plot(m, chi2)
plt.xlabel("m")
plt.ylabel("$\\chi^2 (m)$")

plt.show()