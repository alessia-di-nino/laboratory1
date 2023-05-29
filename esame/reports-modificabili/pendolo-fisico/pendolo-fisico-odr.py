import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr

# Dati---mettete le vostre misure!
# Qui potete anche leggere i dati da file, usando il metodo np.loadtxt(),
# se lo trovate comodo.

#importare Periodi dei vari punti


Tm = []
sigma_T = []

#Creazione array T e sigma_T (periodi e sigma)
for i in range(0,8,1):
    T = np.genfromtxt(
    fname = 'C:\\Users\\hidan\\Desktop\\Università\\Laboratorio\\Relazioni\\I semestre\\Pendolo\\T_punti(esclusi EF).txt', delimiter = '', usecols = (i), unpack = True)
    n = len(T)
    sT = (np.std(T, ddof=1)/np.sqrt(n))/10
    mean_T = (np.mean(T))/10
    Tm = np.append(Tm, [mean_T])
    sigma_T = np.append(sigma_T, [sT])


#Aggiungiamo le due misurazioni diverse

Te = np.genfromtxt(
    fname = 'C:\\Users\\hidan\\Desktop\\Università\\Laboratorio\\Relazioni\\I semestre\\Pendolo\\punto_E.txt', delimiter = '', unpack = True, usecols =(0))

Tf = np.genfromtxt(
    fname = 'C:\\Users\\hidan\\Desktop\\Università\\Laboratorio\\Relazioni\\I semestre\\Pendolo\\punto_F.txt', delimiter = '', unpack = True, usecols = (0))

#divido per 3 e per 5 perché sono state fatte rispettivamente 3 e 5 oscillazioni
sigma_T = np.append(sigma_T, [(np.std(Te,ddof=1)/np.sqrt(len(Te)))/3])
sigma_T = np.append(sigma_T, [(np.std(Tf,ddof=1)/np.sqrt(len(Tf)))/5])

Tm = np.append(Tm, [(np.mean(Te))/3])
Tm = np.append(Tm, [(np.mean(Tf))/5])

y = Tm
sigma_y = sigma_T
#importarte Distanza dei vari punti dal centro di massa

x, sigma_x = np.genfromtxt(
    fname = 'C:\\Users\\hidan\\Desktop\\Università\\Laboratorio\\Relazioni\\I semestre\\Pendolo\\distanza_CentroMassa.txt',
    delimiter = '', unpack = True
    )

n = len(x)

#stampa la media dei periodi e le corrispondenti distanze
print(f'Tm={y}')
#print(f'sigma={sigma_T}')
print(f'Distanze={x}')
#print(sigma_d)

#abbiamo preso come sigma_d la metà della traslazione dei punti rispetto al centro di massa


#conversione in m
x = x/1000
sigma_x = sigma_x/1000



# Definizione dell'accelerazione di gravita‘.
g = 9.761

# Modello per il periodo del pendolo.

##Fit con ODR
#definisco il modello ODR (modello di esempio)
def modello(pars, x):
    return 2.0 * np.pi * np.sqrt((pars[0]**2.0 / 12.0 + x**2.0) / (g * x))
model = odr.Model(modello)      # ovviamente è prima necessario definire un modello (ricordarsi anche di importare la libreria di scipy per odr)
data = odr.RealData(x, y, sx = sigma_x, sy = sigma_y)
alg = odr.ODR(data, model, beta0 = (1.0, 1.0))
out = alg.run()
a_hat = out.beta
sigma_a = np.sqrt(out.cov_beta.diagonal())
chi = out.sum_square

p = np.array([a_hat])    #Questo è necessario se si vuole rappresentare poi il grafico 

#grafico
fig = plt.figure("Nome immagine")
plt.subplot(211)
plt.errorbar(x, y, sigma_x, sigma_y, fmt = 'r.', ecolor = 'black', ms = 4)
plt.grid(which='both', ls='dashed', color='gray')
plt.ylabel('T [s]')
plt.xlim(0.0,0.5)                       # i limiti lungo x vanno impostati a posteriori (impostarli uguali in grafico e residui per una maggiore chiarezza)
#plt.ylim(0.025,0.125)
xx = np.linspace(0.015, 0.50, 50)             # reimpostare i limiti del linspace
plt.plot(xx, modello(a_hat, xx))
#residui
plt.subplot(212)
res = y - modello(a_hat, x)
plt.xlabel('distanza sospensione dal centro di massa [m]')
plt.ylabel('residui [s]')
plt.xlim(0.0,0.5)
plt.errorbar(x, res, sigma_y, fmt = '.', ms = 4)
plt.grid(which='both', ls='dashed', color='gray')
plt.show()


chi_atteso = n - 1      #inserire numero parametri
devstd_chi = np.sqrt(2*chi_atteso)
print(f'Chiquadro effettivo = {chi: .3f}')
print(f'Chiquadro atteso = {chi_atteso: .3f} +/- {devstd_chi: .3f}' )


## p-value
import scipy.stats
def p_value(chisq, ndof):       # chiquadro effettivo, gradi di libertà (ovvero chiquadro atteso)
    p = scipy.stats.chi2.cdf(chisq, ndof)
    if p > 0.5:     #se la probabilità è maggiore del 50% allora prende la probabilità complementare
        p = 1 - p
    return p
print("Il p_value è: ", p_value(chi, chi_atteso))     #immetti il chi effettivo e quello atteso, restituisce la probabilità che sia più estremo di quello ottenuto (ipotizzando vero il modello)


"""
# Scatter plot dei dati.
fig = plt.figure("Pendolo Fisico")
fig.add_axes((0.1, 0.3, 0.8, 0.6))

plt.errorbar(d, Tm, sigma_T, sigma_d, fmt='r.', ecolor = "black")
# Fit---notate che questo e‘ un fit ad un solo parametro.
popt, pcov = curve_fit(period_model, d, Tm, sigma=sigma_T)
l_hat = popt[0]
sigma_l = np.sqrt(pcov[0, 0])
# Confrontate i parametri di best fit con la vostra misura diretta!
print(l_hat, sigma_l)
# Grafico del modello di best-fit.
x = np.linspace(0.015, 0.5, 1000)
plt.plot(x, period_model(x, l_hat))
plt.xlabel('distanza dal centro di massa [m]')
plt.ylabel('Periodo [s]')
plt.xlim((0.001,0.5))
plt.grid(which='both', ls='dashed', color='gray')
plt.savefig('Periodo_lunghezza.pdf')

# Grafico dei residui.
fig.add_axes((0.1, 0.1, 0.8, 0.2))

res = Tm - period_model(d,l_hat)
plt.errorbar(d, res, sigma_T, fmt='.')
plt.grid(which='both', ls='dashed', color='gray')
plt.xlabel('distanza dal centro di massa [m]')
plt.ylabel('Residuals')
plt.ylim(-1.2, 0.5)
plt.xlim(0.001,0.5)
plt.show()
"""