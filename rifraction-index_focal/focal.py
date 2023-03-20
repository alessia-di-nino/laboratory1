import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


#fit lineare
def f(x, m, q):
    return m*x + q

#dati
p = np.array([6.8, 8.8, 10.0, 11.5, 12.8, 13.7, 14.0, 14.5, 15.0, 15.5])/100
q = np.array([9.70, 17.0, 18.0, 26.0, 33.7, 40.5, 44.5, 52.5, 58.0, 60.3])/100
dp = np.full(10, 0.5)/100
dq = 0.03*q
x = 1/p
y = 1/q
dx = dp/(p**2)
dy = dq/(q**2)

# Run a first least-square fit (disregard dx).
popt, pcov = curve_fit(f, x, y, (1., 1.), dy)

# Iteratively update the errors and refit.
for i in range(3):
    dxy = np.sqrt(dy**2. + (popt[0]*dx)**2.)
    popt, pcov = curve_fit(f, x, y, popt, dxy)
    chisq = (((y - f(x, *popt))/dxy)**2.).sum()

# Print the fit output.
print("Passo %d..." % i)
print(popt)
print(np.sqrt(pcov.diagonal()))
print(chisq)

#Graphic
fig = plt.figure('Grafico 1/q-1/p')
fig.add_axes((0.1, .3, 0.85, .6))
xx_fit = np.linspace(np.min(x), np.max(x), 1000)
yy_fit = f(xx_fit, popt[0], popt[1])
plt.plot(xx_fit, yy_fit, color= 'orange', label="Best fit")
plt.ylabel('1/q [m^-1]')
plt.grid(ls='dashed')
plt.errorbar(x, y, dy, dx, c="blue", fmt=".", label="Points taken")
plt.legend(loc="upper right")

#Residuals
fig.add_axes((0.1, .1, 0.85, .2))
res = y - f(x, *popt)
plt.errorbar(x, res, dxy, fmt=".", color="blue")
plt.grid(which="both", color="gray", ls="dashed")
plt.xlabel('1/p [m^-1]')
plt.ylabel("Residui")
plt.ylim(-2, 2)
plt.show()