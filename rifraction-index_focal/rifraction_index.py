import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

# misure di y1 e y2 in quadretti
file_path = "./rifraction-index_focal/data_index.txt"
y1, y2 = np.loadtxt(file_path, delimiter= None, usecols=(0,1), unpack= True)

sigma_y2 = 1 # incertezza in quadretti
sigma_y1 = 1 # incertezza in quadretti


def line(x, n, q):
    """Modello lineare di fit.
    """
    return n * x + q

#grafico: frame 1 superiore contenente il fit
fig1 = plt.figure(1)
frame1=fig1.add_axes((.1,.35,.8,.6))
plt.ylabel("Distanza asse - raggio incidente",fontsize=10)
plt.errorbar(y2, y1, sigma_y1, sigma_y2, fmt=".")
popt, pcov = curve_fit(line, y2, y1)
n_hat, q_hat = popt
sigma_n, sigma_q = np.sqrt(pcov.diagonal())
print(n_hat, sigma_n, q_hat, sigma_q)

x = np.linspace(np.min(y1), np.max(y2), 100)
plt.plot(x, line(x, n_hat, q_hat))
plt.xlabel("Distanza asse - raggio rifratto")
plt.ylabel("Distanza asse - raggio incidente")
plt.grid(ls="dashed", which="both", color="gray")

#frame 2 inferiore contenente i residui
frame2=fig1.add_axes((.1,.1,.8,.2))
frame2.set_ylabel('Residui')
plt.xlabel('Distanza asse - raggio rifratto',fontsize=10)
res = y1 - line(y2, n_hat, q_hat)
plt.errorbar(y2, res, sigma_y1, fmt='.')
plt.axhline(0, color="black")
plt.grid(which="both", ls="dashed", color="gray")

plt.savefig('./fall/fit_res.pdf')

chisq = np.sum(((y1 - line(y2, n_hat, q_hat))/sigma_y1)**2)
print(f'Chi quadro = {chisq :.1f}')

plt.show()








