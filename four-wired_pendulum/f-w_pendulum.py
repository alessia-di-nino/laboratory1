import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


#lettura file
file_path = './four-wired_pendulum/data1.txt'
t, T, Tr = np.loadtxt(file_path, unpack= True)

#def geometria
w = 0.022
d = 1.22
l = 1.123
g = 9.81

#calcolo angolo
v =(w/Tr)*(l/d)
print ('%f', (v))
Theta = np.arccos(1.0 - (v**2)/(2*g*l))
print ('%f', (Theta))

def f_v(x, v0, tau):
    return v0*np.exp(-x/tau)

def f_Theta(x, p1):
    return 2*np.pi*np.sqrt(l/g)*(1+ p1*(x**2)) + 0.01

popt_v, pcov_v = curve_fit(f_v, t, v, np.array([500., 100.]))
v0_fit, tau_fit = popt_v
dv0_fit, dtau_fit = np.sqrt(pcov_v.diagonal())
chisq_v = (((v - f_v(t, v0_fit, tau_fit))**2./f_v(t, v0_fit, tau_fit))).sum()
print ('v0 = %f +- %f m/s' % (v0_fit, dv0_fit))
print ('tau = %f +- %f s' % (tau_fit, dtau_fit))
print (chisq_v)

Theta_0 = np.arccos(1.0 - (v0_fit**2)/(2*g*l))
chisq_T = (((T - f_Theta(Theta, 0.))**2./f_Theta(Theta, 1/16))).sum()
print(Theta_0)
print (chisq_T)


#grafico1
fig = plt.figure('v in funzione del tempo')

plt.ylabel('velocità [m/s]')
plt.grid(color='gray')
plt.plot(t, v, '+', t, f_v(t, v0_fit, tau_fit))
#grafico dei residui1
fig.add_axes((0.1247, 0.01, 0.7755, 0.12))
res = v - f_v(t, v0_fit, tau_fit)
plt.errorbar(t, res, dv0_fit, fmt='+', capsize=5)
plt.grid(which='both', ls='dashed', color='gray')
plt.ylabel('Residuals')
plt.xlabel('Tempo[s]')
plt.ylim(-0.02, 0.02)


#grafico2
fig = plt.figure('Angolo in funzione del tempo')
plt.xlabel('Angolo [rad]')
plt.ylabel('Periodo [s]')
plt.grid(color='gray')
plt.plot(Theta, T, '+', Theta, f_Theta(Theta, 1./16), 'r' )
#grafico dei residui2
fig.add_axes((0.1247, 0.0000000000000001, 0.7755, 0.12))
res = T - f_Theta(Theta, 1./16)
plt.errorbar(Theta, res, 0.0007, fmt='+')
plt.grid(which='both', ls='dashed', color='gray')
plt.ylabel('Residuals')
plt.xlabel('Tempo[s]')
plt.ylim(-0.005, 0.005)


plt.show()
