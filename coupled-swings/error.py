import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

file_path = "./coupled-swings/single-simple.txt"
xA, yA = np.loadtxt(file_path, delimiter=None, skiprows=0, usecols = [0,1], unpack=True)
n = len(xA)

yA = yA - np.full(yA.shape, yA[0])
plt.figure("A")
plt.errorbar(xA, yA, fmt=".")
plt.xlabel("tempo [s]")
plt.ylabel("ampiezza [u.a.]")
plt.grid(which="both", ls="dashed", color="gray")

sigmax= np.std(xA, ddof=1)
print(f"sigma_xA = {sigmax}")
print(n)

plt.show()