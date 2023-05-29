import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

#lista delle distanze del punto di sospensione dal centro di massa
d = np.array([0.42, 0.32, 0.22, 0.12, 0.02, 0.08, 0.18, 0.28, 0.38, 0.48])
sigma_d = np.full(d.shape, 0.001)

#matrice (10 x 6) contenente sei prese dati per ognuno dei 10 fori da cui ho lasciato oscillare il pendolo
misure = np.loadtxt("./physical_pendulum/dati.txt")
print(misure)

T = []
sigma_T = []

def calculate_sigma(riga : np.array) -> float: 
    risultato_sommatoria = 0
    for elemento in riga:
        risultato_sommatoria += (elemento - np.mean(riga))**2 

    return np.sqrt( 1/ (len(riga)* (len(riga)-1))  *   risultato_sommatoria ) #formula della deviazione standard della media

for riga in misure:
    T.append( np.mean(riga) / 10  ) #perchè per ognuna delle sei misurazioni del periodo, si sono fatte dieci oscillazioni (quindi dieci periodi)
    sigma_T.append( calculate_sigma(riga) )

print(T, sigma_T)

g = 9.81

def period_model(d, l):
    return 2.0 * np.pi * np.sqrt((l**2 / 12.0 + d**2.0) / (g * d))

plt.figure("Periodo")
fig = plt.figure("Periodo")
fig.add_axes((0.1, 0.3, 0.8, 0.6))
plt.errorbar(d, T, sigma_T, sigma_d, fmt=".")
popt, pcov = curve_fit(period_model, d, T, sigma=sigma_T)
l_hat = popt[0]
sigma_l = np.sqrt(pcov[0, 0])
print(l_hat, sigma_l)

x = np.linspace(np.min(d), np.max(d), 100)
plt.plot(x, period_model(x, l_hat))
plt.xlabel("d [m]")
plt.ylabel("Periodo [s]")
plt.grid(which="both", ls="dashed", color="gray")

fig.add_axes((0.1, 0.1, 0.8, 0.2))

res = T - period_model(d, l_hat)
plt.errorbar(d, res, sigma_T, fmt=".")
plt.axhline(0, color="black")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Distanza [cm]")
plt.ylabel("Residui")

plt.savefig("./physical_pendulum/fit and residuals.pdf")

chisq = np.sum(((T - period_model(d, l_hat))/sigma_T)**2)
print(f'Chi quadro = {chisq :.1f}')

#calcolo chi quadro
chi2 = ((((T - period_model(x, l_hat))) / sigma_T)**2).sum()
dof = 8 - 2
print(f'chi2 del fit: {chi2} / {dof} dof')
def p_value(chi2, dof):
    p = scipy.stats.chi2.cdf(chi2, dof)
    if p > 0.5:
        p = 1.0 - p
    return p
print(p_value(chi2, 6))

plt.show()