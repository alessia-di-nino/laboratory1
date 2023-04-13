import numpy as np
import matplotlib
from matplotlib import pyplot as plt

file_path = "./geometric_optics/rainbow/rainbow.png"
plt.figure("Immagine originale")
img = matplotlib.image.imread(file_path)
plt.xlabel("x [pixels]")
plt.ylabel("y [pixels]")
plt.imshow(img)
plt.savefig("./geometric_optics/rainbow/immagine-originale.pdf")

file_path = "./geometric_optics/rainbow/points_in.txt"
x, y = np.loadtxt(file_path, delimiter=None, skiprows=0, usecols=(0,1), unpack=True)
fig = plt.figure('fit circolare interno')

plt.errorbar(x, y, fmt='.')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim([150, 1100])
plt.ylim([150, 1100])
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
plt.savefig("./geometric_optics/rainbow/fit-in.pdf")

file_path1 = "./geometric_optics/rainbow/points_ext.txt"
x1, y1 = np.loadtxt(file_path1, delimiter=None, skiprows=0, usecols=(0,1), unpack=True)
fig = plt.figure('fit circolare esterno')

plt.errorbar(x1, y1, fmt='.')
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim([100, 1150])
plt.ylim([50, 1150])
plt.grid()
plt.gca().set_aspect("equal") #serve a forzare la griglia quadrata (rispetta le proporzioni tra assi)

def fit_circle(x1, y1, sigma1):
    n1 = len(x1)
    x_m1 = np.mean(x1)
    y_m1 = np.mean(y1)
    u1 = x1 - x_m1
    v1 = y1-y_m1

    s_u1 = np.sum(u1)
    s_uu1 = np.sum (u1**2.0)
    s_uuu1 = np.sum(u1**3.0)
    s_v1= np.sum(v1)
    s_vv1 = np.sum(v1**2.0)
    s_vvv1 = np.sum(v1**3.0)
    s_uv1 = np.sum(u1 * v1)
    s_uuv1 = np.sum(u1 * u1 * v1)
    s_uvv1 = np. sum (u1* v1 * v1)
    D1 = 2.0*(s_uu1 * s_vv1 - s_uv1**2.0)

    u_c1 = (s_vv1 * (s_uuu1 + s_uvv1) - s_uv1*(s_vvv1 + s_uuv1))/ D1
    v_c1 = (s_uu1 * (s_vvv1 + s_uuv1) - s_uv1*(s_uuu1 + s_uvv1)) / D1
    x_c1 = u_c1 + x_m1
    y_c1 = v_c1 + y_m1
    r1 = np.sqrt(u_c1**2.0 + v_c1**2.0 + (s_uu1 + s_vv1) / n1)
    sigma_xy1 = sigma1 * np.sqrt(2.0 / n1)
    sigma_r1 = sigma1 * np.sqrt(1.0/n1)
    return x_c1, y_c1, r1, sigma_xy1, sigma_r1

np.random.seed (1)
sigma1 = 0.05
x_c1, y_c1, r1, sigma_xy1, sigma_r1 = fit_circle(x1, y1, sigma1)
print (f'x_c1 = {x_c1:.3f} +/- {sigma_xy1:.3f}')
print (f'y_c1 = {y_c1:.3f} +/- {sigma_xy1:.3f}')
print (f'r1 = {r1: .3f} +/- {sigma_r1:.3f}')
x1 = np.linspace(400, 900, 353)
y1 = np.linspace(200, 600, 353)
theta1 = np.linspace(0.0, 2.0*np.pi, 353)
a1 = x_c1 + r1 * np.cos(theta1)
b1 = y_c1 + r1 * np.sin(theta1)
plt.plot(a1, b1)
plt.savefig("./geometric_optics/rainbow/fit-ext.pdf")

plt.show()
