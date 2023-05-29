import numpy as np

T = [2.132710, 2.132646, 2.132637, 2.132642, 2.132633, 2.132582, 2.132535, 2.132532]

media = np.mean(T)
n = 8

dev = np.std(T)

print(dev, media)

''' sono stati presi 10 periodi grazie al programma di acquisizione, perciò si può stimare a posteriori l'errore sul periodo con la deviazione standard dalla media. Si nota quindi che gli errori sulle x (theta) e quelli sulle y (periodi) sono comparabili, dunque si usa il pacchetto odr per realizzare il grafico periodo-ampiezza. '''

a =  #m
sigma_a = 0.005 #m
l = 1.097 #m
sigma_l = 0.005 #m

theta = np.arcsin((a)/(l))
sigma_theta = np.sqrt( (sigma_a**2)/(1 - (a**2)/(l**2)) + (sigma_l**2)/((l**4)*(1 - (a**2)/(l**2)) ) )