import math as mt
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats
import sys
sys.path.append('/home/marco/Desktop/Uni_anno1/Laboratorio1/')
from LaTex import *

path='/home/marco/Desktop/Uni_anno1/Laboratorio1/Relazione/'
data='masse_periodi.csv'

m, s_m, T, s_T = np.loadtxt(path+data, delimiter=',', unpack=True)

m=m*0.001
s_m=s_m*0.001

#DoubleArrayTable(path+'tabella_periodi', T, s_T)


##modello

def model1(m, M, k):
    return 2*np.pi*np.sqrt((m + M)/k)

#20 periodi
#T_20= [10.10]

##massa

'''
m_s= 7.773
d_s= 0.001

m_i= np.array([5.006, 10.011, 20.010, 40.042, 50.044, 55.054 ])
s_m_i= np.full(m_i.shape, 0.001)

m= (m_i + m_s)*0.001
s_m= np.full(m.shape, np.sqrt(2*(0.001**2)))*(0.001/mt.sqrt(12))
'''
#grammi la massa da 100g è inutilizzabile in quanto non sta nel supporto e la molla allungata a causa sua sbatte contro il supporto

##periodi, 20 periodi per massa

'''
T_20= np.matrix([[10.10, 10.10, 10.16, 10.15, 10.19],
                [11.69, 11.91, 11.82, 11.66, 11.66],
                [14.25, 14.22, 14.13, 14.07, 14.06],
                [18.19, 18.13, 18.31, 18.12, 18.22],
                [19.84, 19.88, 19.78, 19.84, 19.82],
                [20.59, 20.62, 20.68, 20.50, 20.56]])

s_T20= np.full(T_20.shape, 0.01)

T1_20= T_20/20
s1_T20= s_T20/20

print(T1_20, '\n', s1_T20, '\n')   #qui ci siamo

T= np.array([])
s_T= np.array([])

for i in range(0, 6):
    T= np.append(T, np.mean(T1_20[i]))
    s_T= np.append(s_T, (np.std(T1_20[i], ddof=1))/mt.sqrt(5))


#print(T, '\n', s_T, '\n') #anche qui ci siamo
'''


## posizioni di equilibrio per ogni massa

y_eq= np.array([28.5, 27.8, 27.0, 26.2, 25.5, 24.8, 24.0])*0.01#metro a natro, centimetri
s_yeq= np.full(y_eq.shape, 0.1/np.sqrt(12))*0.01#metro a nastro pm 1mm

DoubleArrayTable(path+'tabella_conf_masse_yeq', m, s_m, y_eq, s_yeq)

##fit dei minimi quadrati di test

popt, pcov= curve_fit( model1, m, T, p0=(0.076, 14.6), sigma= s_T, absolute_sigma=True)
M_hat1, k_hat1 = popt
s_M, s_k = np.sqrt(pcov.diagonal())

chisq= (((T - model1(m, *popt))/(s_T))**2).sum()

print(f'Massa: {M_hat1} +/- {s_M}', '\n', f'Costante elastica: {k_hat1} +/- {s_k}', '\n', f'chi quadro: {chisq}')

res5 = T - model1(m, *popt)



#plt.savefig(path+'fit_relazione.png')


##controllo delle incertezze

c= abs((np.pi/np.sqrt(k_hat1))*(1/np.sqrt(m + (M_hat1/3))))*s_m

a5=np.array([], 'bool')

mask1 = (c - s_T*0.1 > 0)

print('Controllo= ', sum(mask1), '\n')

##odr per periodi

def Period(pars, m): return 2*np.pi*np.sqrt((m + pars[0])/pars[1])

from scipy import odr

model= odr.Model(Period)
data= odr.RealData(m, T, sx= s_m, sy= s_T )

#beta0 sono i valori iniziali
alg= odr.ODR(data, model, beta0=(0.076, 14.6))

out= alg.run()

M_hat, k_hat = out.beta

s_M1, s_k1 = np.sqrt(out.cov_beta.diagonal())

chisq1= out.sum_square

print(f'Massa: {M_hat} +/- {s_M1}', '\n', f'Costante elastica: {k_hat} +/- {s_k1}', '\n', f'chi quadro: {chisq1}')

P = 2*np.pi*np.sqrt((m + M_hat)/k_hat)

res = T - P

s_eff1 = np.sqrt(s_T**2 + (np.pi**2 * s_m**2)/(k_hat*(m + M_hat)))



##minimi quadrati con funzione invertita
def model2(T, M, k): return ((k * T**2)/(4*np.pi**2)) - M

popt1, pcov1 = curve_fit(model2, T, m, p0=(0.076, 14.6), sigma= s_m, absolute_sigma=True)

M_hat2, k_hat2 = popt1
s_M2, s_k2 = np.sqrt(pcov1.diagonal())

res2 = m - model2(T, *popt1)

chisq2 = np.sum((res2**2)/(s_m**2))

print(f'Massa: {M_hat2} +/- {s_M2}', '\n', f'Costante elastica: {k_hat2} +/- {s_k2}', '\n', f'chi quadro: {chisq2}')

fig = plt.figure('scatter-plot, residui', figsize=(10, 6), dpi=100)

ax1, ax2= fig.subplots(2, 1, sharex=True, gridspec_kw= dict(height_ratios=[2, 1], hspace=0.05))

ax1.errorbar(m, T, s_T, s_m, fmt='.', label='Dati')
ax1.plot(m, P, label='Modello')
ax1.set_ylabel('T[s]')
ax1.grid(color='lightgray', ls='dashed')
ax1.legend()

ax2.errorbar(m, res, s_eff1, fmt='.')
ax2.grid(ls='dashed', color='lightgray')
ax2.set_xlabel('massa [g]')
ax2.set_ylabel('Residuals [s]')
ax2.plot(m, np.full(m.shape, 0.0))

fig.align_ylabels((ax1, ax2))
plt.ylim(-0.005, 0.005)
#plt.savefig(path+'fit_periodi2.png', dpi=300)


## fit legge di hooke

def hooke(pars, m): return m*pars[0] + pars[1]

from scipy import odr

model= odr.Model(hooke)
data= odr.RealData(m, y_eq, sx= s_m, sy= s_yeq )

#beta0 sono i valori iniziali
alg= odr.ODR(data, model, beta0=(0.805, 14.9))

out= alg.run()

gk_hat, l0_hat = out.beta

s_gk, s_l0 = np.sqrt(out.cov_beta.diagonal())

chisq4= out.sum_square

s_g= np.sqrt((s_k/k_hat1)**2+(s_gk/gk_hat)**2)

print(f'l0_hat= {l0_hat} +/- {s_l0}', '\n' , f'gk_hat= {gk_hat} +/- {s_gk}', '\n', f'Chi quadro: {chisq4}', '\n', f'g= {k_hat1*gk_hat} +/- {s_g}')

M = m*gk_hat + l0_hat

res = y_eq - M

s_eff = np.sqrt(s_yeq**2 + (gk_hat**2) * (s_m**2))

fig1=plt.figure('Fit lineare con Hooke', figsize=(10,6), dpi=100)

ax3, ax4= fig1.subplots(2,1, sharex=True, gridspec_kw= dict(height_ratios=[2, 1], hspace=0.05))
ax3.errorbar(m, y_eq, s_yeq, s_m, fmt='.', label='datas')
ax3.plot(m, M, label='model')
ax3.grid(ls='dashed', color='lightgray')
ax3.set_ylabel('Masse [kg]')
ax3.legend()

ax4.errorbar(m, res, s_eff, fmt='.')
ax4.plot(m, np.full(m.shape, 0.0))
ax4.grid(ls='dashed', color='lightgray')
ax4.set_xlabel('masa [kg]')
ax4.set_ylabel('posizione di equilirio [m]')

fig.align_ylabels((ax3, ax4))
#plt.savefig(path+'fit_pos_equil.png', dpi=300)

plt.show()

##secondo controllo sugli errori

c1= abs(gk_hat)*s_yeq

mask2 = (c1 - s_m*0.1 > 0)

print('Controllo= ', sum(mask2), '\n')

## p-value

def p_value (chisq, ndof):
    p= scipy.stats.chi2.cdf(chisq, ndof)

    if p > 0.5:
        p = 1.0 - p

    return p

print('p-vlaue= ', p_value(chisq1, 5))
print(f'p-value = {p_value(chisq4, 5)}')


plt.show()








