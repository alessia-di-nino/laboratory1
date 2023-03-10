import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
 
#creazione degli array di numpy in cui inserire i file listati (poco efficace,
# perchè se bisogna riordinare in qualche modo, è probabile che si perda l'associazione; meglio a oggetti)
 
m = np.array([8.360, 11.893, 18.904, 24.861, 44.885])
sigma_m = np.full(m.shape, 0.001)
m = m/1000
sigma_m = sigma_m/1000
 
d = np.array([12.80, 14.35, 16.70, 18.30, 22.25])
sigma_d = np.full(d.shape, 0.05)
d = d/1000
sigma_d = sigma_d/1000
 
r = d / 2.0
sigma_r = sigma_d / 2.0
 
V = 4.0 / 3.0 * np.pi * r**3.0
sigma_V = V * 3.0 * sigma_d / d
 
# definizione dei fit (retta e legge di potenza)
 
def line(x, m, q):
    return m * x + q
 
def power_law(x, norm, index):
    return norm * (x**index) #norm = k, costante di normalizzazione, index=
    #esponente della variabile indipendente (raggio)
 
    #fit retta
 
plt.figure("Grafico Massa_Volume")
plt.errorbar(m, V, sigma_V, sigma_m, fmt=".")
popt, pcov = curve_fit(line, m, V)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(m_hat, sigma_m, q_hat, sigma_q)
 
x = np.linspace(0, np.max(m), 5)
plt.plot(x, line(x, m_hat, q_hat))
plt.xlabel("Massa [Kg]")
plt.ylabel("Volume [m$^3$]")
plt.grid(which = "both", ls="dashed", color="gray")
plt.savefig("./density/steel/Mass_Volume.pdf")
 
    #fit legge di potenza
 
plt.figure("Grafico Raggio_Massa")
plt.errorbar(r, m, sigma_m, sigma_r, fmt=".")
popt, pcov = curve_fit(power_law, r, m)
norm_hat, index_hat = popt
sigma_norm, sigma_index = np.sqrt(pcov.diagonal())
print(norm_hat, sigma_norm, index_hat, sigma_index)
 
x = np.linspace(np.min(r), np.max(r), 5) #inserire 3 al posto di np.min(r) per vedere l'intercetta con r=1
plt.plot(x, power_law(x, norm_hat, index_hat))
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Raggio [m]")
plt.ylabel("Massa [Kg]")
plt.grid(which="both", ls="dashed", color="gray")
plt.axhline(0, color="black")
plt.savefig("./density/steel/Radium_Mass.pdf")
 
#residui
 
plt.figure("Grafico dei residui")
res = V - line(m, m_hat, q_hat)
plt.errorbar(m, res, sigma_V, fmt=".")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Massa [Kg]")
plt.ylabel("Residui")
plt.axhline(0, color="black")
plt.savefig("./density/steel/steel_Residuals.pdf")
 
plt.show()

