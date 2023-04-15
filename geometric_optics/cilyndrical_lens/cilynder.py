import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


#fit lineare
def f(x, m, q):
    return m*x + q

#dati
p = np.array([-6.5, -7.4, -7.8, -8.0, -9.6, -12.4, -20.5])/100
q = np.array([43.5, 37.3, 29.4, 23.2, 17.7, 12.9, 8.3])/100
dp = np.full(7, 0.5)/100
dq = 0.05*q
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
plt.plot(xx_fit, yy_fit, color= 'orange')
plt.ylabel('1/q [m^-1]')
plt.grid(ls='dashed')
plt.errorbar(x, y, dy, dx, c="blue", fmt=".")
plt.legend(loc="upper right")

#Residuals
fig.add_axes((0.1, .1, 0.85, .2))
res = y - f(x, *popt)
plt.errorbar(x, res, dxy, fmt=".", color="blue")
plt.grid(which="both", color="gray", ls="dashed")
plt.xlabel('1/p [m^-1]')
plt.ylabel("Residui")
plt.ylim(-2, 2)
plt.savefig("./geometric_optics/cilyndrical_lens/fit")

plt.show()