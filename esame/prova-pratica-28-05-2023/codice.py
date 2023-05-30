import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
 
#programmazione di oggetti per definire ogni classe di oggetto (parallelepipedo, cilindro, prisma)
 
'''class Parallelepipedo:
    def __init__(self, base : float, altezza : float, massa : float):
        self.base = base
        self.altezza = altezza
        self.massa = massa
        self.volume = self.base**2 * self.altezza
   
    def get_err_vol(self, sigma_misure : float) ->float:
        return self.volume * np.sqrt((4 * sigma_misure**2 / self.base**2 ) + sigma_misure**2 / self.altezza**2)'''
 
   
class Cilindro:
    def __init__(self, diametro : float, altezza : float, massa : float):
        self.diametro = diametro
        self.raggio = self.diametro/2
        self.altezza = altezza
        self.massa = massa
        self.volume =  np.pi * self.raggio**2 * self.altezza
 
    def get_err_vol(self, sigma_altezza : float, sigma_diametro : float) ->float:
        return  self.volume * np.sqrt((4 * sigma_diametro**2 / self.raggio**2 ) + sigma_altezza**2 / self.altezza**2)
 
class Prisma:
    def __init__(self, base : float, altezza : float, massa : float):
        self.base = base
        self.altezza = altezza
        self.massa = massa
        self.volume = base**2 * np.tan(60) * 3 * altezza
 
    def get_err_vol(self, sigma_altezza : float, sigma_diametro : float) -> float:
        return self.volume * np.sqrt((4 * sigma_diametro**2 / self.base**2 ) + sigma_altezza**2 / self.altezza**2)
 
#creazione della lista di oggetti che dovranno rappresentare i punti sul grafico
 
solidi = []
 
#solidi.append( Parallelepipedo( base = 9.90 , altezza = 41.48, massa = 32.738))
solidi.append( Cilindro(diametro = 6.46/1000, altezza = 95.38/1000, massa = 22.388/1000))
solidi.append( Cilindro(diametro = 10.82/1000, altezza = 37.50/1000, massa = 24.605/1000))
solidi.append( Cilindro(diametro = 5.96/1000, altezza = 36.70/1000, massa = 8.572/1000))
solidi.append( Cilindro(diametro = 10.46/1000, altezza = 16.00/1000, massa = 10.468/1000))
solidi.append( Prisma( base = 15.45/1000, altezza = 17.62/1000, massa = 28.598/1000))
 
masse = [s.massa for s in solidi]
volumi = [s.volume for s in solidi]
errori_vol = [s.get_err_vol(0.01/(1000 * np.sqrt(12)), 0.02/(1000 * np.sqrt(12))) for s in solidi]
errori_masse = [0.001/(1000 * np.sqrt(12)) for i in range(len(volumi))]
print(volumi)
print(errori_vol)

#fit
 
def line(x, m, q):
    return m * x + q
 
plt.figure("Grafico Massa_Volume")
plt.errorbar(masse, volumi, errori_vol, errori_masse, fmt=".")
popt, pcov = curve_fit(line, masse, volumi)
m_hat, q_hat = popt
sigma_m, sigma_q = np.sqrt(pcov.diagonal())
print(f" m_hat: {m_hat}, sigma_m: {sigma_m}, q_hat: {q_hat}, sigma_q: {sigma_q}")
 
x = np.linspace(np.min(masse), np.max(masse), 5)
plt.plot(x, line(x, m_hat, q_hat))
plt.xlabel("Massa [Kg]")
plt.ylabel("Volume [m$^3$]")
plt.grid(which = "both", ls="dashed", color="gray")
plt.savefig("./density/brass/Mass_Volume.pdf")
 
#residui
 
plt.figure("Grafico dei residui")
res = volumi - line(np.array(masse), m_hat, q_hat)
plt.errorbar(masse, res, errori_vol, fmt=".")
plt.grid(which="both", ls="dashed", color="gray")
plt.xlabel("Massa [Kg]")
plt.ylabel("Residui")
plt.axhline(0, color="black")
plt.savefig("./density/brass/brass_Residuals.pdf")

V_magg = (34.921/1000)*m_hat
print(f"volume del solido maggiore = {V_magg}")
sigma_Vmagg = V_magg * np.sqrt((0.001/34.921)**2 + (sigma_m/m_hat)**2)
print(f"errore sul volume del solido maggiore = {sigma_Vmagg}")

V_dim = (41.78/1000) * (1.00/1000)**2
sigma_Vdim = V_dim * np.sqrt((0.02/41.78)**2 + (2*0.02/1.00)**2)
print(f"volume del solido di dimensioni note = {V_dim}")
print(f"errore sul volume del solido di dimensioni note = {sigma_Vdim}")

chisq = np.sum(((volumi - line(x, m_hat, q_hat))/errori_vol)**2)
print(f'Chi quadro = {chisq :.1f}')
      
plt.show()
 
 

