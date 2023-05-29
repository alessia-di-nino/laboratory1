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

# Calculate the square of the period and propagate the errors.
T2 = T**2.
sigma_T2 = 2 * T * sigma_T

def line(x, m, q):
    """Funzione di fit (una semplice retta).
    """
    return m * x + q

# best fit
popt, pcov = curve_fit(line, m, T2)
m_fit, q_fit = popt
sigma_m_fit, sigma_q_fit = np.sqrt(pcov.diagonal())
print(m_fit, sigma_m_fit, q_fit, sigma_q_fit)

#grafico periodo quadro massa
fig = plt.figure('Grafico periodo quadro-massa')
fig.add_axes((0.1, 0.3, 0.8, 0.6))
plt.errorbar(m, T2, sigma_T2, sigma_m, fmt='o')

x = np.linspace(0., 60., 100)
plt.plot(x, line(x, m_fit, q_fit))
plt.xlabel('Massa [g]')
plt.ylabel('Periodo al quadrato [s$^2$]')
plt.grid(ls='dashed')
plt.xlim(0,60)
plt.gca().set_xticklabels([]) #questo toglie i numerini sulle x


#residui
x = m
fig.add_axes((0.1, 0.1, 0.8, 0.2))
res = T2 - line(x, m_fit, q_fit)
plt.errorbar(x, res, sigma_m, fmt="o")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel('Massa [g]')
plt.ylabel("Residui")
plt.ylim(-0.007, 0.007)
plt.xlim(0, 60)

#calcolo chi quadro
chi2 = (((T2 - line(x, m_fit, q_fit)) / sigma_T2)**2).sum()
dof = 4 - 2
print(f'chi2 del fit: {chi2} / {dof} dof')
def p_value(chi2, dof):
    p = scipy.stats.chi2.cdf(chi2, dof)
    if p > 0.5:
        p = 1.0 - p
    return p
print(p_value(chi2, 2))

plt.show()