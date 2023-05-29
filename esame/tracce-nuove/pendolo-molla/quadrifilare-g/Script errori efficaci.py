import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def f(x, g):
    y = 2.0 * np.pi * np.sqrt(1.08 / g) * (1.0 + (x**2.0/16.0) + x**4.0 * (11.0/3072.0))
    return y

# Dati Casuali
x = np.array([0.0748, 0.1496, 0.2244, 0.2992, 0.3739, 0.4488, 0.5236]) #theta
rand = np.random.uniform(-0.1, 0.1, len(x))
y = f(x, 9.81) + rand #periodo

sig_x = np.full(x.shape, 0.002)
sig_y = np.full(y.shape, 0.00000028)

# Primo fit "rozzo"
popt, pcov = curve_fit(f, x, y, sigma=sig_y)
pcov = np.sqrt(pcov.diagonal())
g_hat = popt[0]

# Metodo degli errori efficaci
for i in range(3):
        sig_eff = np.sqrt(sig_y**2.0 + ( 2.0 * np.pi * np.sqrt( (1.08 / g_hat) * ((x / 8.0) + ((44.0 * x**3.0) / 3072.0))) )**2.0 )
        popt, pcov = curve_fit(f, x, y, sigma=sig_eff)


print(popt[0])
print(np.sqrt(pcov.diagonal()))

