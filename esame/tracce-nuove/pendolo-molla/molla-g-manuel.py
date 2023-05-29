import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

L = np.array([2850, 2780, 2700, 2620, 2550, 2480, 2400])#lunghezze di eq, la lunghezza a riposo la fitto e confronto
sigma_L = np.array([3, 3, 3, 3, 3, 3, 3]) #da valutare
T = np.array([6982, 6770, 6529, 6270, 6055, 5780, 5540]) #/5 misure di 5 periodi, 3 volte ciascuno, media
sigma_T = np.array([1, 1, 1, 1, 1, 1, 1]) #5 periodi, 3 volte ciascuno, media e std
M = np.array([754, 652, 560, 460, 377, 278, 189]) #misure masse
sigma_M = np.array([1, 1, 1, 1, 1, 1, 1])/np.sqrt(12)
m = 1  #misure molla


def rad(M, k, C):
    return 2*np.pi*np.sqrt((M + m/3)/k) + C



popt, pcov = curve_fit(rad, M, T, sigma = sigma_T)
k_hat, C_hat = popt
sigma_k, sigma_c = np.sqrt(pcov.diagonal())
fig = plt.figure('Grafico di Fit', figsize=(7.,7.))
fig.add_axes((0.15, 0.35, 0.8, 0.6))
plt.errorbar(M, T, sigma_T, sigma_M, fmt='.')
plt.plot(M, rad(M, k_hat, C_hat))
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Masse[Kg]')
plt.ylabel('Periodi[s]')

fig.add_axes((0.15, 0.1, 0.8, 0.18))
res = T - rad(M, k_hat, C_hat)
plt.errorbar(M, res, yerr=sigma_T, fmt='.')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Masse[Kg]')
plt.ylabel('Residuals')

chi = 0
for i in range (0, len(T)):
    chi = chi + ((T[i] - rad(M[i], k_hat, C_hat))/sigma_T[i])**2

print(k_hat, sigma_k, C_hat)
print(chi)
plt.show()


def fin(M, g, Q):
    return M*g/(k_hat) + Q

popt, pcov = curve_fit(fin, M, L, sigma = sigma_L)
g_hat, Q_hat = popt
sigma_g = np.sqrt(pcov.diagonal())

fig1 = plt.figure('Grafico di Fit', figsize=(7.,7.))
fig1.add_axes((0.15, 0.35, 0.8, 0.6))
plt.errorbar(M, L, yerr=sigma_L, xerr=sigma_M, fmt='.')
plt.plot(M, fin(M, g_hat, Q_hat))
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Masse[Kg]')
plt.ylabel('Pos. di eq.[m]')

fig1.add_axes((0.15, 0.1, 0.8, 0.18))
res1 = L - fin(M, g_hat, Q_hat)
plt.errorbar(M, res1, yerr=sigma_L, fmt='.')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Masse[Kg]')
plt.ylabel('Residuals')

chi1 = 0
for i in range (0, len(L)):
    chi1 = chi1 + ((L[i] - fin(M[i], g_hat, Q_hat))/sigma_L[i])**2

print(chi1)
print(g_hat, sigma_g)

plt.show()