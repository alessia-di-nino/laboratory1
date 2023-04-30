import numpy as np
import matplotlib
from matplotlib import pyplot as plt

file_path = "./geometric_optics/moon_halo/halo.png"
plt.figure("Immagine originale")
img = matplotlib.image.imread(file_path)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.imshow(img)
plt.savefig("./geometric_optics/moon_halo/immagine-originale.pdf")

file_path = "./geometric_optics/moon_halo/points.txt"
x, y = np.loadtxt(file_path, delimiter=None, skiprows=0, usecols=(0,1), unpack=True)
fig = plt.figure('fit circolare')

plt.errorbar(x, y, fmt='.')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim([400, 850])
plt.ylim([150, 600])
plt.grid()
plt.gca().set_aspect("equal") #serve a forzare la griglia quadrata (rispetta le proporzioni tra assi)

def fit_circle(x, y, sigma):
    n = len(x)
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y-y_m

    s_u = np.sum(u)
    s_uu = np.sum (u**2.0)
    s_uuu = np.sum(u**3.0)
    s_v= np.sum(v)
    s_vv = np.sum(v**2.0)
    s_vvv = np.sum(v**3.0)
    s_uv = np.sum(u * v)
    s_uuv = np.sum(u * u * v)
    s_uvv = np. sum (u* v * v)
    D = 2.0*(s_uu * s_vv - s_uv**2.0)

    u_c = (s_vv * (s_uuu + s_uvv) - s_uv*(s_vvv + s_uuv))/ D
    v_c = (s_uu * (s_vvv + s_uuv) - s_uv*(s_uuu + s_uvv)) / D
    x_c = u_c + x_m
    y_c = v_c + y_m
    r = np.sqrt(u_c**2.0 + v_c**2.0 + (s_uu + s_vv) / n)
    sigma_xy = sigma * np.sqrt(2.0 / n)
    sigma_r = sigma * np.sqrt(1.0/n)
    return x_c, y_c, r, sigma_xy, sigma_r

np.random.seed (1)
sigma = 0.05
x, y = np.loadtxt(file_path, delimiter=None, skiprows=0, usecols=(0,1), unpack=True)
x_c, y_c, r, sigma_xy, sigma_r = fit_circle(x, y, sigma)
print (f'x_c = {x_c:.3f} +/- {sigma_xy:.3f}')
print (f'y_c = {y_c:.3f} +/- {sigma_xy:.3f}')
print (f'r = {r: .3f} +/- {sigma_r:.3f}')
x = np.linspace(400, 900, 353)
y = np.linspace(200, 600, 353)
theta = np.linspace(0.0, 2.0*np.pi, 353)
a = x_c + r * np.cos(theta)
b = y_c + r * np.sin(theta)
plt.plot(a, b)
plt.savefig("./geometric_optics/moon_halo/fit.pdf")

if (x_c < x and y_c < y):
    PHI=  np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

elif x_c > x and y_c < y:
    PHI=  np.pi/2 + np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

elif x_c > x and y_c > y:
    PHI=  np.pi + np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

else: 
    PHI=  3*np.pi/2 + np.arcsin(np.abs(y-y_c)/(np.sqrt((x_c - x)**2 + (y_c - y)**2)))

fig2= plt.figure('Residui_alone_lunare')
res= r - np.sqrt((x_c - x)**2 + (y_c - y)**2)
plt.errorbar(PHI, res, sigma, fmt='.', color='tomato')
plt.plot(PHI, np.full(PHI.shape, 0.0), color='chartreuse')
plt.grid(color='lightgray', ls='dashed')
plt.xlabel('Angolo campionamenti-orizzontale [$\circ$]')
plt.ylabel('Residui [pixel]')
plt.savefig("RES")

plt.show()
