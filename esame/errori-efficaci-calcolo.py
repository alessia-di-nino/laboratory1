import numpy as np
from scipy.optimize import curve_fit

def fit_model(x, m, q):
    return m*x + q

file path = "..."
x, sigma_x, y, sigma_y = np.loadtxt(file_path, unpack=True)
popt, pcov = curve_fit(fit_model, x, y, sigma=sigma_y)

for i in range(3):
    sigma_eff = np.sqrt(sigma_y**2.0 + (popt[0] * sigma_x)**2.0)
    popt, pcov = curve_fit(fit_model, x, y, sigma = sigma_eff)
    chisq = (((y - fit_model(x, *popt))/sigma_eff)**2.0).sum()

    print(f"step{i}...")
    print(popt, np.sqrt(pcov.diagonal()))
    print(f"Chisquare = {chisq:.2f}")