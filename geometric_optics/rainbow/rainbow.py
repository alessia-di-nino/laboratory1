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

file_path2 = "./geometric_optics/rainbow/red-points.txt"
x2, y2 = np.loadtxt(file_path2, delimiter=None, skiprows=0, usecols=(0,1), unpack=True)

file_path3 = "./geometric_optics/rainbow/yellow-points.txt"
x3, y3 = np.loadtxt(file_path3, delimiter=None, skiprows=0, usecols=(0,1), unpack=True)

fig = plt.figure('bande colorate')

colors = []
for i in range(len(x3)):
    colors.append("yellow")

for i in range(len(x2)):
    colors.append("red")

plt.scatter( np.concatenate([x3, x2]), np.concatenate( [y3 , y2]), c=colors, marker=".")
plt.grid(which="both", ls="dashed", color="gray")
plt.savefig("./geometric_optics/rainbow/colors.pdf")

fig = plt.figure('fit banda rossa')

plt.errorbar(x2, y2, fmt='.', color="red")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim([100, 1150])
plt.ylim([50, 1150])
plt.grid()
plt.gca().set_aspect("equal") #serve a forzare la griglia quadrata (rispetta le proporzioni tra assi)

def fit_circle(x2, y2, sigma2):
    n2 = len(x2)
    x_m2 = np.mean(x2)
    y_m2 = np.mean(y2)
    u2 = x2 - x_m2
    v2 = y2-y_m2

    s_u2 = np.sum(u2)
    s_uu2 = np.sum (u2**2.0)
    s_uuu2 = np.sum(u2**3.0)
    s_v2= np.sum(v2)
    s_vv2 = np.sum(v2**2.0)
    s_vvv2 = np.sum(v2**3.0)
    s_uv2 = np.sum(u2 * v2)
    s_uuv2 = np.sum(u2 * u2 * v2)
    s_uvv2 = np. sum (u2* v2 * v2)
    D2 = 2.0*(s_uu2 * s_vv2 - s_uv2**2.0)

    u_c2 = (s_vv2 * (s_uuu2 + s_uvv2) - s_uv2*(s_vvv2 + s_uuv2))/ D2
    v_c2 = (s_uu2 * (s_vvv2 + s_uuv2) - s_uv2*(s_uuu2 + s_uvv2)) / D2
    x_c2 = u_c2 + x_m2
    y_c2 = v_c2 + y_m2
    r2 = np.sqrt(u_c2**2.0 + v_c2**2.0 + (s_uu2 + s_vv2) / n2)
    sigma_xy2 = sigma2 * np.sqrt(2.0 / n2)
    sigma_r2 = sigma2 * np.sqrt(1.0/n2)
    return x_c2, y_c2, r2, sigma_xy2, sigma_r2

np.random.seed (1)
sigma2 = 0.05
x_c2, y_c2, r2, sigma_xy2, sigma_r2 = fit_circle(x2, y2, sigma2)
print (f'x_c2 = {x_c2:.3f} +/- {sigma_xy2:.3f}')
print (f'y_c2 = {y_c2:.3f} +/- {sigma_xy2:.3f}')
print (f'r2 = {r2: .3f} +/- {sigma_r2:.3f}')
x2 = np.linspace(400, 900, 353)
y2 = np.linspace(200, 600, 353)
theta2 = np.linspace(0.0, 2.0*np.pi, 353)
a2 = x_c2 + r2 * np.cos(theta2)
b2 = y_c2 + r2 * np.sin(theta2)
plt.plot(a2, b2)
plt.savefig("./geometric_optics/rainbow/red-fit.pdf")

fig = plt.figure('fit banda gialla')

plt.errorbar(x3, y3, fmt='.', color="yellow")
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.xlim([100, 1150])
plt.ylim([50, 1150])
plt.grid()
plt.gca().set_aspect("equal") #serve a forzare la griglia quadrata (rispetta le proporzioni tra assi)

def fit_circle(x3, y3, sigma3):
    n3 = len(x3)
    x_m3 = np.mean(x3)
    y_m3 = np.mean(y3)
    u3 = x3 - x_m3
    v3 = y3-y_m3

    s_u3 = np.sum(u3)
    s_uu3 = np.sum (u3**2.0)
    s_uuu3 = np.sum(u3**3.0)
    s_v3= np.sum(v3)
    s_vv3 = np.sum(v3**2.0)
    s_vvv3 = np.sum(v3**3.0)
    s_uv3 = np.sum(u3 * v3)
    s_uuv3 = np.sum(u3 * u3 * v3)
    s_uvv3 = np. sum (u3* v3 * v3)
    D3 = 2.0*(s_uu3 * s_vv3 - s_uv3**2.0)

    u_c3 = (s_vv3 * (s_uuu3 + s_uvv3) - s_uv3*(s_vvv3 + s_uuv3))/ D3
    v_c3 = (s_uu3 * (s_vvv3 + s_uuv3) - s_uv3*(s_uuu3 + s_uvv3)) / D3
    x_c3 = u_c3 + x_m3
    y_c3 = v_c3 + y_m3
    r3 = np.sqrt(u_c3**2.0 + v_c3**2.0 + (s_uu3 + s_vv3) / n3)
    sigma_xy3 = sigma3 * np.sqrt(2.0 / n3)
    sigma_r3 = sigma3 * np.sqrt(1.0/n3)
    return x_c3, y_c3, r3, sigma_xy3, sigma_r3

np.random.seed (1)
sigma3 = 0.05
x_c3, y_c3, r3, sigma_xy3, sigma_r3 = fit_circle(x3, y3, sigma3)
print (f'x_c3 = {x_c3:.3f} +/- {sigma_xy3:.3f}')
print (f'y_c3 = {y_c3:.3f} +/- {sigma_xy3:.3f}')
print (f'r3 = {r3: .3f} +/- {sigma_r3:.3f}')
x3 = np.linspace(400, 900, 353)
y3 = np.linspace(200, 600, 353)
theta3 = np.linspace(0.0, 2.0*np.pi, 353)
a3 = x_c3 + r3 * np.cos(theta3)
b3 = y_c2 + r3 * np.sin(theta3)
plt.plot(a3, b3)
plt.savefig("./geometric_optics/rainbow/yellow-fit.pdf")
plt.show()
