import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

#funzione
def oscillazione(t, A, tao, w, phi, C):
    """Modello di oscillazione.
    """
    return A*np.exp(-t/tao)*np.cos(w*t + phi) + C

#dati
file_path = "./coupled-swings/single.txt"
t, y = np.loadtxt(file_path, delimiter=None, usecols=(2,3), unpack=True)
sigma_y = 1  # [u.a.]

# Grafico principale.
fig = plt.figure('Fit e residui pendolo singolo')
fig.add_axes((0.1, 0.3, 0.8, 0.6))
plt.errorbar(t, y, sigma_y, fmt='b.')
popt, pcov = curve_fit(oscillazione, t, y, p0=(300, 33, 4.427, 0.7, 12))
A_hat, tao_hat, w_hat, phi_hat, C_hat = popt
sigma_A, sigma_tao, sigma_w, sigma_phi, sigma_C = np.sqrt(pcov.diagonal())
print("\nParametro A: A = ", A_hat, "\nIncertezza su A: sigma_A = ", sigma_A, "\n\nParametro tao:  tao = ", tao_hat, "\nIncertezza su tao: sigma_tao = ", sigma_tao,"\n\nParametro w:  w = ", w_hat, "\nIncertezza su w: sigma_w = ", sigma_w, "\n\nParametro phi:  phi = ", phi_hat, "\nIncertezza su phi: sigma_phi = ", sigma_phi, "\n\nParametro C:  C = ", C_hat, "\nIncertezza su C: sigma_C = ", sigma_C)
plt.plot(t, oscillazione(t, A_hat, tao_hat, w_hat, phi_hat, C_hat), "r")
plt.grid(which='both', ls='dashed', color='gray')
plt.ylabel('x [cm]')

# Grafico dei residui.
fig.add_axes((0.1, 0.1, 0.8, 0.2))
res = y - oscillazione(t, A_hat, tao_hat, w_hat, phi_hat, C_hat)
plt.errorbar(t, res, sigma_y, fmt='b.')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('t [s]')
plt.ylabel('Residuals')
plt.ylim(-4.0, 4.0)

# Calcolo del chi quadro.
chi2 = (((y - oscillazione(t, A_hat, tao_hat, w_hat, phi_hat, C_hat)) / sigma_y)**2).sum()
dof = len(t) - 5
print(f'\nchi^2 del fit: {chi2} / {dof} dof')

plt.show()
