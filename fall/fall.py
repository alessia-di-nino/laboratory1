import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

t = np.loadtxt("fall_time.txt")
h = np.loadtxt("fall_height")
sigma_h = #da calcolare con un ciclo for

h = h / 100.0
sigma_h = sigma_h/100.0

#fit quadratico

def parabola(t, a, v0, h0)
    return 0.5 * a * t**2.0 + v0 * t + h0

plt.figure("legge oraria")
plt.errorbar(t, h, sigma_h, fmt=".")
popt, pcov = curve_fit(parabola, t, h, sigma=sigma_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))
print(a_hat, sigma_a, v0_hat, sigma_v0, h0_hat, sigma_h0)

x = np.linspace(0.0, 0.7, 100)
plt.plot(x, parabola(x, *popt))
plt.xlabel("Tempo [s]")
plt.ylabel("Altezza [m]")
plt.grid(ls="dashed", which="both", color="gray")
plt.savefig("legge_oraria.pdf")

plt.show()