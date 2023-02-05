import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

t = np.loadtxt("./fall/fall_time.txt")
h = np.loadtxt("./fall/fall_height.txt")
blur_h = np.loadtxt("./fall/fall_blur-height.txt")

blur_h = (blur_h * np.sqrt(2)) / 200.0
h = h / 100.0

#fit quadratico

def parabola(t, a, v0, h0):
    return 0.5 * a * t**2.0 + v0 * t + h0

plt.figure("legge oraria")
plt.errorbar(t, h, blur_h, fmt=".")
popt, pcov = curve_fit(parabola, t, h, sigma=blur_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))
print(a_hat, sigma_a, v0_hat, sigma_v0, h0_hat, sigma_h0)

x = np.linspace(np.min(t), np.max(t), 100)
plt.plot(x, parabola(x, *popt))
plt.xlabel("Tempo [s]")
plt.ylabel("Altezza [m]")
plt.grid(ls="dashed", which="both", color="gray")
plt.savefig("./fall/legge_oraria.pdf")

plt.show()