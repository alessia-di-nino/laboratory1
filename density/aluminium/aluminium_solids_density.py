import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
 
#programmazione di oggetti per definire ogni classe di oggetto (parallelepipedo, cilindro, prisma)
 
class Parallelepipedo:
    def __init__(self, base : float, altezza : float, profondità : float, massa : float):
        self.base = base
        self.altezza = altezza
        self.profondità = profondità
        self.massa = massa
        self.volume = self.base * self.profondità * self.altezza
   
    def get_err_vol(self, sigma_misure : float) ->float:
        return self.volume * np.sqrt((sigma_misure**2 / self.base**2 ) + (sigma_misure**2 / self.altezza**2) + (sigma_misure**2 / self.profondità**2))
 
   
class Cilindro:
    def __init__(self, diametro : float, altezza : float, massa : float):
        self.diametro = diametro
        self.raggio = self.diametro/2
        self.altezza = altezza
        self.massa = massa
        self.volume =  np.pi * self.raggio**2 * self.altezza
 
    def get_err_vol(self, sigma_misure : float) ->float:
        return  self.volume * np.sqrt((4 * sigma_misure**2 / self.raggio**2 ) + sigma_misure**2 / self.altezza**2)
 
#creazione della lista di oggetti che dovranno rappresentare i punti sul grafico
 
solidi = []
 
solidi.append( Parallelepipedo( base = 10.06, altezza = 17.65, profondità = 10.00, massa = 4.773 ) )
solidi.append( Cilindro(diametro = 11.96, altezza = 19.14, massa = 5.790) )
solidi.append( Cilindro(diametro=5.95, altezza=19.42, massa=1.457))
 
masse = [s.massa for s in solidi]
volumi = [s.volume for s in solidi]
errori_vol = [s.get_err_vol(0.01) for s in solidi]
errori_masse = [0.001 for i in range(len(volumi))]
 
#fit
 
def line(x, m, q):
    return m * x + q
 
plt.figure("Grafico Massa-Volume")
plt.errorbar(masse, volumi, errori_vol, errori_masse, fmt=".")
popt, pcov = curve_fit(line, masse, volumi)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(f" m_hat: {m_hat}, sigma_m: {sigma_m}, q_hat: {q_hat}, sigma_q: {sigma_q}")
 
x = np.linspace(0, np.max(masse), 3)
plt.plot(x, line(x, m_hat, q_hat))
plt.xlabel("Massa [Kg]")
plt.ylabel("Volume [m$^3$]")
plt.grid(which = "both", ls="dashed", color="gray")
plt.savefig("./density/aluminium/Mass_Volume.pdf")
 
#residui
 
plt.figure("Grafico dei residui")
res = volumi - line(np.array(masse), m_hat, q_hat)
plt.errorbar(masse, res, errori_vol, fmt=".")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Massa [Kg]")
plt.ylabel("Residui")
plt.axhline(0, color="black")
plt.savefig("./density/aluminium/aluminium_Residuals.pdf")
 
plt.show()

