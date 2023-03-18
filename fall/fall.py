import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

t = np.loadtxt("./fall/fall_time.txt")
h = np.loadtxt("./fall/fall_height.txt")
blur_h = np.loadtxt("./fall/fall_blur-height.txt")

blur_h = (blur_h * np.sqrt(2)) / 200.0
h = h / 100.0

print(blur_h)
#fit quadratico

def parabola(t, a, v0, h0):
    return 0.5 * a * t**2.0 + v0 * t + h0

'''
plt.figure("legge oraria")
fig = plt.figure("legge oraria")
plt.errorbar(t, h, blur_h, fmt=".")
popt, pcov = curve_fit(parabola, t, h, sigma=blur_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))
print(a_hat, sigma_a, v0_hat, sigma_v0, h0_hat, sigma_h0)

x = np.linspace(np.min(t), np.max(t), 100)
plt.plot(x, parabola(x, *popt))
plt.xlabel("Tempo [s]")
plt.ylabel("Altezza [m]")
plt.grid(ls="dashed", which="both", color="gray")

fig.add_axes((0.1, 0.1, 0.8, 0.2))

res = h - parabola(t, a_hat, v0_hat, h0_hat)
plt.errorbar(t, res, blur_h, fmt=".")
plt.axhline(0, color="black")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Tempo [s]")
plt.ylabel("Residui")
'''

fig1 = plt.figure(1)
frame1=fig1.add_axes((.1,.35,.8,.6))
plt.ylabel('Altezza [m]',fontsize=10)
plt.errorbar(t, h, blur_h, fmt=".")
popt, pcov = curve_fit(parabola, t, h, sigma=blur_h)
a_hat, v0_hat, h0_hat = popt
sigma_a, sigma_v0, sigma_h0 = np.sqrt(np.diagonal(pcov))
print(a_hat, sigma_a, v0_hat, sigma_v0, h0_hat, sigma_h0)

x = np.linspace(np.min(t), np.max(t), 100)
plt.plot(x, parabola(x, *popt))
plt.xlabel("Tempo [s]")
plt.ylabel("Altezza [m]")
plt.grid(ls="dashed", which="both", color="gray")

#Parte inferiore contenente i residui
frame2=fig1.add_axes((.1,.1,.8,.2))
frame2.set_ylabel('Residui')
plt.xlabel('Tempo [s]',fontsize=10)
res = h - parabola(t, a_hat, v0_hat, h0_hat)
plt.errorbar(t, res, blur_h, fmt=".")
plt.axhline(0, color="black")
plt.grid(which="both", ls="dashed", color="gray")

plt.savefig('./fall/fit_res.pdf')

chisq = np.sum(((h - parabola(t, a_hat, v0_hat, h0_hat))/blur_h)**2)
print(f'Chi quadro = {chisq :.1f}')

plt.show()