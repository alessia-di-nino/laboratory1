import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

x1 = np.array([6.5, 8.7, 10.8, 13.1, 15.2, 17.3, 19.4, 21.6, 23.7, 25.9, 28.1, 30.2, 32.3, 34.5, 36.7, 38.8, 41.0, 43.1, 45.2, 47.3])
x=x1/100
sigma_x = np.full(x.shape, 0.1)

T = np.array([31.0, 30.0, 29.5, 29.0, 28.5, 28.0, 27.5, 27.0, 26.5, 26.0, 25.5, 25.0, 24.5, 24.0, 23.5, 23.5, 22.5, 22.0, 21.5, 21.0])
sigma_T = np.full(T.shape, 0.1)

def line(x, m, q):
    return m * x + q

plt.figure("Grafico posizione-temperatura")
plt.errorbar(x, T, sigma_T, sigma_x, fmt=".")

popt, pcov = curve_fit(line, x, T, sigma=sigma_T)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(m_hat, sigma_m, q_hat, sigma_q)
2
xx = np.linspace(np.min(x),np.max(x), 1000)
plt.plot(xx, line(xx, m_hat, q_hat))

plt.xlabel("Posizione [cm]")
plt.ylabel("Temperatura [$^\\circ$C]")
plt.grid(which="both", ls = "dashed", color = "gray")
plt.savefig("posizione_temperatura.pdf")

plt.savefig("residui-rame.pdf")
plt.show()



#per attaccare un errore a una misura è importante capire cosa sto misurando e da cosa dipende