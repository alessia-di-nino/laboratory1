import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
 
#creazione degli array di numpy in cui inserire i file listati (poco efficace,
# perchè se bisogna riordinare in qualche modo, è probabile che si perda l'associazione; meglio a oggetti)
 
m = np.array([8.360, 11.893, 18.904, 24.861, 44.885])
sigma_m = np.full(m.shape, 0.001)
 
d = np.array([12.80, 14.35, 16.70, 18.30, 22.25])
sigma_d = np.full(d.shape, 0.05)
 
r = d / 2.0
sigma_r = sigma_d / 2.0
 
V = 4.0 / 3.0 * np.pi * r**3.0
sigma_V = V * 3.0 * sigma_d / d
 
# definizione dei fit (retta e legge di potenza)
 
def line(x, m, q):
    return m * x + q
 
def power_law(x, norm, index):
    return norm * (x**index)
 
    #fit retta
 
plt.figure("Grafico Massa_Volume")
plt.errorbar(m, V, sigma_V, sigma_m, fmt=".")
popt, pcov = curve_fit(line, m, V)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(m_hat, sigma_m, q_hat, sigma_q)
 
x = np.linspace(0, np.max(m), 5)
plt.plot(x, line(x, m_hat, q_hat))
plt.xlabel("Massa [g]")
plt.ylabel("Volume [mm$^3$]")
plt.grid(which = "both", ls="dashed", color="gray")
plt.savefig("./density/steel/Mass_Volume.pdf")
 
    #fit legge di potenza
 
plt.figure("Grafico Massa_Raggio")
plt.errorbar(m, r, sigma_r, sigma_m, fmt=".")
popt, pcov = curve_fit(power_law, m, r)
norm_hat, index_hat = popt
sigma_norm, sigma_index = np.sqrt(pcov.diagonal())
print(norm_hat, sigma_norm, index_hat, sigma_index)
 
x = np.linspace(np.min(m), np.max(m), 5)
plt.plot(x, power_law(x, norm_hat, index_hat))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Massa [g]")
plt.ylabel("Raggio [mm]")
plt.grid(which="both", ls="dashed", color="gray")
plt.savefig("./density/steel/Mass_Radium.pdf")
 
#residui
 
plt.figure("Grafico dei residui")
res = V - line(m, m_hat, q_hat)
plt.errorbar(m, res, sigma_V, fmt=".")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Massa [g]")
plt.ylabel("Residui")
plt.axhline(0, color="black")
plt.savefig("./density/steel/steel_Residuals.pdf")
 
plt.show()

