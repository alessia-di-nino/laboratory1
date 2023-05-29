import numpy as np
from scipy import odr
import matplotlib.pyplot as plt

def fit_model(pars, x):
    # Note the independent variable is the last argument.
    return pars[0] * x + pars[1]

# Read the data from file.
x, y = np.loadtxt("./geometric_optics/rifraction-index_focal/data_index.txt", unpack=True)
dx= np.full(x.shape, 1) #quadretti
dy= np.full(y.shape, 1) #quadretii
# Run the actual ODR.
model = odr.Model(fit_model)
data = odr.RealData(x, y, sx=dx, sy=dy)
alg = odr.ODR(data, model, beta0=(1.0, 1.0))
out = alg.run()
m_hat, q_hat = out.beta
sigma_m, sigma_q = np.sqrt(out.cov_beta.diagonal())
chisq = out.sum_square
# Print the fit output.
print(f'm = {m_hat:.3f} +/- {sigma_m:.3f}')
print(f'q = {q_hat:.3f} +/- {sigma_q:.3f}')
print(f'Chisquare = {chisq:.1f}')

na_hat= 1/m_hat
dna= sigma_m/((m_hat)**2)


print('INDICE DI RIFRAZIONE:', na_hat, dna, 'INTERCETTA:', q_hat, sigma_q)
err_rel_n = dna/na_hat
print('errore relativo', err_rel_n)


#residui 
res= np.sqrt(out.delta**2 + out.eps**2)


print(res)

res= np.array([0.41246276, 0.31298341, 0.00473664, 0.28286957, 0.38810028, 0.33734589, 0.25275574, 0.41821412, 0.53088867, 0.50280427])
#grafico
fig=plt.figure('Rifrazione Plexiglass odr', figsize=(10., 6.), dpi=100)
ax1, ax2 = fig.subplots(2, 1, sharex=True, gridspec_kw=dict(height_ratios=[2, 1], hspace=0.05))


ax1.errorbar(x, y, dy, dx, fmt='.', label='punti sperimentali', color='darkslateblue')
ax1.plot(x, fit_model(out.beta, x), label='Modello di best-fit', color='lightsteelblue')
ax1.set_ylabel('Distanza normale - raggio incidente [quadretti]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()
ax2.errorbar(x, res, dy, fmt='.',  color='darkslateblue')
ax2.plot(x, np.full(x.shape, 0.0), color='lightsteelblue')
ax2.set_xlabel('Distanza normale - raggio rifratto [quadretti]')
ax2.set_ylabel('Residui [a. u.]')
ax2.grid(color='lightgray', ls='dashed')
#plt.ylim(-4.0, 4.0)
fig.align_ylabels((ax1, ax2))
plt.show()