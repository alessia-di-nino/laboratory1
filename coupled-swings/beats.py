import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#funzione 1

def oscillazione1(t, A1, tao1, w1, phi1, C1):
    """Modello di oscillazione.
    """
    return A1*np.exp(-t/tao1)*np.cos(w1*t + phi1) + C1

#dati 1
file_path = "./coupled-swings/beats.txt"
t, y = np.loadtxt(file_path, delimiter=None, usecols=(2,3), unpack=True)
sigma_y = 1   # [u.a.]

# Grafico 1.
fig = plt.figure('Fit1 e residui1')
fig.add_axes((0.1, 0.3, 0.8, 0.6))
popt, pcov = curve_fit(oscillazione1, t, y, p0=(-100, 33, 4.427, 0.0883, 4))
A1_hat, tao1_hat, w1_hat, phi1_hat, C1_hat = popt
sigma_A1, sigma_tao1, sigma_w1, sigma_phi1, sigma_C1 = np.sqrt(pcov.diagonal())
print("\nParametro A1: A1 = ", A1_hat, "\nIncertezza su A1: sigma_A1 = ", sigma_A1, "\n\nParametro tao1:  tao1 = ", tao1_hat, "\nIncertezza su tao1: sigma_tao1 = ", sigma_tao1,"\n\nParametro w1:  w1 = ", w1_hat, "\nIncertezza su w1: sigma_w1 = ", sigma_w1, "\n\nParametro phi1:  phi1 = ", phi1_hat, "\nIncertezza su phi1: sigma_phi1 = ", sigma_phi1, "\n\nParametro C1:  C1 = ", C1_hat, "\nIncertezza su C1: sigma_C1 = ", sigma_C1)
plt.errorbar(t, y)
plt.grid(which='both', ls='dashed', color='gray')
plt.ylabel('x [cm]')

#funzione 2
def oscillazione2(t, A2, tao2, w2, phi2, C2):
    """Modello di oscillazione.
    """
    return A2*np.exp(-t/tao2)*np.cos(w2*t + phi2) + C2

#dati 2
file_path = "./coupled-swings/beats.txt"
t, y2 = np.loadtxt(file_path, delimiter=None, usecols=(0,1), unpack=True)
sigma_y = 1   # [u.a.]

# Grafico 2.
popt, pcov = curve_fit(oscillazione2, t, y2, p0=(-100, 33, 4.427, -1, 5))
A2_hat, tao2_hat, w2_hat, phi2_hat, C2_hat = popt
sigma_A2, sigma_tao2, sigma_w2, sigma_phi2, sigma_C2 = np.sqrt(pcov.diagonal())
print("\nParametro A2: A2 = ", A2_hat, "\nIncertezza su A2: sigma_A2 = ", sigma_A2, "\n\nParametro tao2:  tao2 = ", tao2_hat, "\nIncertezza su tao2: sigma_tao2 = ", sigma_tao2,"\n\nParametro w2:  w2 = ", w2_hat, "\nIncertezza su w2: sigma_w2 = ", sigma_w2, "\n\nParametro phi2:  phi2 = ", phi2_hat, "\nIncertezza su phi2: sigma_phi2 = ", sigma_phi2, "\n\nParametro C2:  C2 = ", C2_hat, "\nIncertezza su C2: sigma_C2 = ", sigma_C2)
plt.errorbar(t, y2)
plt.grid(which='both', ls='dashed', color='gray')
plt.ylabel('x [cm]')

# Grafici dei residui 1
fig.add_axes((0.1, 0.1, 0.8, 0.2))
res1 = y - oscillazione1(t, A1_hat, tao1_hat, w1_hat, phi1_hat, C1_hat)
plt.errorbar(t, res1, sigma_y, fmt='r.')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('t [s]')
plt.ylabel('Residuals')
plt.ylim(-10.0, 10.0)


# Grafico dei residui 2
res2 = y2 - oscillazione2(t, A2_hat, tao2_hat, w2_hat, phi2_hat, C2_hat)
plt.errorbar(t, res2, sigma_y, fmt='g.')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('t [s]')
plt.ylabel('Residuals')
plt.ylim(-10.0, 10.0)


plt.show()