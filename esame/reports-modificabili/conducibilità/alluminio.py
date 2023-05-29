import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

x1 = np.array([7.0, 9.5, 12.0, 14.0, 17.0, 19.0, 22.0, 24.5, 27.0, 29.5, 32.0, 34.0, 37.0, 39.5, 42.0])
x=x1/100
sigma_x = np.full(x.shape, 0.1)/100

T = np.array([40.0, 39.0, 38.0, 36.0, 35.0, 34.0, 33.0, 31.0, 30.0, 28.0, 27.0, 26.0, 25.0, 23.0, 22.0])
sigma_T = np.full(T.shape, 0.2)

def line(x, m, q):
    return m * x + q

plt.figure("Grafico posizione-temperatura")
plt.errorbar(x, T, sigma_T, sigma_x, fmt=".")

popt, pcov = curve_fit(line, x, T, sigma=sigma_T)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(m_hat, sigma_m, q_hat, sigma_q)

xx = np.linspace(np.min(x), np.max(x), 1000)
plt.plot(xx, line(xx, m_hat, q_hat))

plt.xlabel("Posizione [cm]")
plt.ylabel("Temperatura [$^\\circ$C]")
plt.grid(which="both", ls = "dashed", color = "gray")
plt.savefig("posizione_temperatura.pdf")


plt.show()