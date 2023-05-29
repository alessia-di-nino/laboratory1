import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats

# Misura di allungamento e periodo di oscillazione in funzione della massa appesa.
m = np.array([5.005, 10.006, 20.011, 50.032])
sigma_m = np.array([0.001] * len(m))
l = np.array([152., 171., 220., 351.])
sigma_l = np.array([1.] * len(l))
T = np.array([0.530, 0.611, 0.739, 1.044])
sigma_T = np.array([0.005, 0.005, 0.006, 0.004])

def line(x, m, q):
    """Funzione di fit (una semplice retta).
    """
    return m * x + q

# best fit
popt, pcov = curve_fit(line, m, l)
m_fit, q_fit = popt
sigma_m_fit, sigma_q_fit = np.sqrt(pcov.diagonal())
print(m_fit, sigma_m_fit, q_fit, sigma_q_fit)

#grafico allungamento massa
fig = plt.figure('Grafico allungamento-massa')
fig.add_axes((0.1, 0.3, 0.8, 0.6))
plt.errorbar(m, l, sigma_l, sigma_m, fmt='o')
x = np.linspace(0., 60., 100)
plt.plot(x, line(x, m_fit, q_fit))
plt.xlabel('Massa [g]')
plt.ylabel('Allungamento [mm]')
plt.grid(ls='dashed')
plt.xlim(0,60)
plt.gca().set_xticklabels([]) #questo toglie i numerini sulle x

#residui allungamento massa
x = m
fig.add_axes((0.1, 0.1, 0.8, 0.2))
res = l - line(x, m_fit, q_fit)
plt.errorbar(x, res, sigma_m, fmt="o")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel('Massa [g]')
plt.ylabel("Residui")
plt.ylim(-4, 4)
plt.xlim(0,60)

#calcolo chi quadro
chi2 = (((l - line(x, m_fit, q_fit)) / sigma_l)**2).sum()
dof = 4 - 2
print(f'chi2 del fit: {chi2} / {dof} dof')
def p_value(chi2, dof):
    p = scipy.stats.chi2.cdf(chi2, dof)
    if p > 0.5:
        p = 1.0 - p
    return p
print(p_value(chi2, 2))

plt.show()