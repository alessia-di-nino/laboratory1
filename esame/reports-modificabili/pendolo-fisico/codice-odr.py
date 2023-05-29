import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

def fit_model(d, l):
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

model = odr.Model(fit_model)
data = odr.RealData(d, misure, sx=sigma_d, sy=sigma_T)
alg = odr.ODR(data, model, beta0=(1.0, 1.0))
out = alg.run()
l_hat, g_hat = out.beta
sigma_l, sigma_g = np.sqrt(out.cov_beta.diagonal())
chisq=out.sum_square

print(f"l = {l_hat:.3f} +/- {sigma_l:.3f}")
print(f"l = {g_hat:.3f} +/- {sigma_g:.3f}")

plt.show()