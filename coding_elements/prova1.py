import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def f(x, omega):
    return np.sin(omega*x)

omega0 = 2.2
n = 10
x = np.linspace(0., 10., n)
y = f(x, omega0)
dy = np.full(y.shape, 0.0)
y += np.random.normal

xgrid =

popt, pcov = curve_fit(f, x, y, sigma=dy)
print(popt, pcov)

plt.figure("Dati")
plt.errorbar(x, y, dy, fmt = ".")
