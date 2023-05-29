import numpy as np
from scipy.odr import odrpack
from matplotlib import pyplot as plt
import scipy.stats

#dati
T = []  #qui ci metto le medie per i 10 periodi dei diversi 5 theta
sigma_T = [] #qui ci vanno le deviazioni standard per i 5 gruppi di periodi
Theta = [] #qui ci vanno i cinque diversi theta
sigma_Theta = np.array([] * len(Theta)) #qui ci vanno gli errori

l = 1.097 #m
sigma_l = 0.005 #m

#funzione per il periodo con parametro l/g e theta variabile
def f_Theta(pars, th):
    return 2*np.pi*np.sqrt(pars[0])*(1+ ((1/16)*(th**2)) + ((11/3072)*(th**4)))

x = Theta
dx = sigma_Theta
y = T
dy = sigma_T

#fit usando odr perchè errore sull'ampiezza non trascurabile
model = odrpack.Model(f_Theta)
data = odrpack.RealData(x, y, sx=dx, sy=dy)
odr = odrpack.ODR(data, model, beta0=np.array([0.11]))
out = odr.run()
omega_hat= out.beta
sigma_omega = (np.sqrt(out.cov_beta.diagonal()))
chisq = out.sum_square
print(omega_hat, sigma_omega)
print(f'Chisquare = {chisq:.1f}')
plt.errorbar(Theta, T, dy, dx, fmt='.')
plt.plot(Theta, f_Theta(omega_hat, Theta))
plt.xlabel('Ampiezza [rad]')
plt.ylabel('Periodo [s]')
plt.show()

#calcolo del chi quadro per 5-1 gradi di libertà
def p_value(chisq, ndof):
    p = scipy.stats.chi2.cdf(chisq, ndof)
    if p > 0.5:
        p = 1.0 - p
    return p

print(p_value(chisq, 4))