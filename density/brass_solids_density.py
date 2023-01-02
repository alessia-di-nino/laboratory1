import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
 
#programmazione di oggetti per definire ogni classe di oggetto (parallelepipedo, cilindro, prisma)
 
class Parallelepipedo:
    def __init__(self, base : float, altezza : float, massa : float):
        self.base = base
        self.altezza = altezza
        self.massa = massa
        self.volume = self.base**2 * self.altezza
   
    def get_err_vol(self, sigma_misure : float) ->float:
        return self.volume * np.sqrt((4 * sigma_misure**2 / self.base**2 ) + sigma_misure**2 / self.altezza**2)
 
   
class Cilindro:
    def __init__(self, diametro : float, altezza : float, massa : float):
        self.diametro = diametro
        self.raggio = self.diametro/2
        self.altezza = altezza
        self.massa = massa
        self.volume =  np.pi * self.raggio**2 * self.altezza
 
    def get_err_vol(self, sigma_misure : float) ->float:
        return  self.volume * np.sqrt((4 * sigma_misure**2 / self.raggio**2 ) + sigma_misure**2 / self.altezza**2)
 
class Prisma:
    def __init__(self, base : float, altezza : float, massa : float):
        self.base = base
        self.altezza = altezza
        self.massa = massa
        self.volume = base**2 * np.tan(60) * 3 * altezza
 
    def get_err_vol(self, sigma_misure : float) -> float:
        return self.volume * 2 * sigma_misure / self.base
 
#creazione della lista di oggetti che dovranno rappresentare i punti sul grafico
 
solidi = []
 
solidi.append( Parallelepipedo( base = 9.90, altezza = 41.48, massa = 32.738))
solidi.append( Cilindro(diametro = 10.12, altezza = 37.58, massa = 24.613))
solidi.append( Cilindro(diametro = 10.08, altezza = 16.08, massa = 10.505))
solidi.append( Prisma( base = 9.74, altezza = 22.20, massa = 16.422))
 
masse = [s.massa for s in solidi]
volumi = [s.volume for s in solidi]
errori_vol = [s.get_err_vol(0.02) for s in solidi]
errori_masse = [0.001 for i in range(len(volumi))]
 
#fit
 
def line(x, m, q):
    return m * x + q
 
plt.figure("Grafico Massa_Volume")
plt.errorbar(masse, volumi, errori_vol, errori_masse, fmt=".")
popt, pcov = curve_fit(line, masse, volumi)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(f" m_hat: {m_hat}, sigma_m: {sigma_m}, q_hat: {q_hat}, sigma_q: {sigma_q}")
 
x = np.linspace(0, np.max(masse), 4)
plt.plot(x, line(x, m_hat, q_hat))
plt.xlabel("Massa [g]")
plt.ylabel("Volume [mm$^3$]")
plt.grid(which = "both", ls="dashed", color="gray")
plt.savefig("Volume_Massa.pdf")
 
#residui
 
plt.figure("Grafico dei residui")
res = volumi - line(np.array(masse), m_hat, q_hat)
plt.errorbar(masse, res, errori_vol, fmt=".")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Massa [g]")
plt.ylabel("Residui")
plt.axhline(0, color="black")
plt.show()
 
 

