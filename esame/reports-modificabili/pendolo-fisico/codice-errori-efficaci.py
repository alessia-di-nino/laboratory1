import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt

g=9.81

def period_model(d, l):
    return 2.0 * np.pi * np.sqrt((l**2 / 12.0 + d**2.0) / (g * d))

d = np.array([0.42, 0.32, 0.22, 0.12, 0.02, 0.08, 0.18, 0.28, 0.38, 0.48])
sigma_d = np.full(d.shape, 0.001)

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

popt, pcov = curve_fit(period_model, d, T, sigma=sigma_T)

for i in range(3):
    sigma_eff = np.sqrt(list(map(lambda x: x**2.0 + (popt[0] * sigma_d)**2.0, sigma_T)))
    popt, pcov = curve_fit(period_model, d, T, sigma = sigma_eff)
    chisq = (((T - period_model(d, *popt))/sigma_eff)**2.0).sum()

    print(f"step{i}...")
    print(popt, np.sqrt(pcov.diagonal()))
    print(f"Chisquare = {chisq:.2f}")