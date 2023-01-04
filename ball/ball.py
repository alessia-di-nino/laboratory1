import wave
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

file_path = './ball/Palla-audio.wav'
stream = wave.open(file_path)
signal = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
if stream.getnchannels() == 2:
    signal = signal[::2]
t = np.arange(len(signal)) / stream.getframerate()

plt.figure('Rimbalzi pallina')
plt.plot(t, signal)
plt.xlabel('Tempo [s]')
plt.ylabel("Registrazione audio")
plt.grid(which='both', ls='dashed', color='gray')
plt.savefig('./ball/audio_rimbalzi.pdf')
plt.show()

t = np.array ([0.660, 1.535, 2.201, 2.717, 3.139, 3.457, 3.734, 3.948, 4.122])
sigma_t = 0.01

# Calcolo delle differenze di tempo.
dt = np.diff(t)

# Creazione dell’array con gli indici dei rimbalzi.
n = np.arange(len(dt)) + 1.

# Calcolo dell’altezza massima e propagazione degli errori.
h = 9.81 * (dt**2.) / 8.0
dh = 2.0 * np.sqrt(2.0) * h * sigma_t / dt

def expo(n, h0, gamma):
    return h0 * gamma**n

plt.figure('Altezza dei rimbalzi')
plt.errorbar(n, h, dh, fmt='.')
popt, pcov = curve_fit(expo, n, h, sigma=dh)
h0_hat, gamma_hat = popt
sigma_h0, sigma_gamma = np.sqrt(pcov.diagonal())
print(h0_hat, sigma_h0, gamma_hat, sigma_gamma)

x = np.linspace(np.min(n), np.max(n), 100)
plt.plot(x, expo(x, h0_hat, gamma_hat))
plt.yscale('log')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('Rimbalzo')
plt.ylabel('Altezza massima [m]')
plt.savefig('./ball/altezza_rimbalzi.pdf')

plt.figure("Residui")
res = h - expo(n, h0_hat, gamma_hat)
plt.errorbar(n, res, sigma_h0, fmt=".")
plt.axhline(0, color="black")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Rimbalzi")
plt.ylabel("Residui")
plt.savefig('./ball/residui.pdf')

plt.show()
