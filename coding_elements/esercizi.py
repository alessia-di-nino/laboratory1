l = [1, 7, 9, 10, 2, 5, 4]

min = l[0]

for i in l:
    if i < l[0]:
        min = i

print(min)

max = l[0]

for i in l:
    if i > max:
        max = i

print(max)

for i in range(len(l)):
    for j in range(len(l) - 1):
        if l[j] > l[j + 1]:
            l[j], l[j + 1] = l[j + 1], l[j]
print(l)

def f(x):
    r = x**3 + 1
    return r

a = -5
b = 5

while (b - a) > 0.1:
    if f(a) * f(b) < 0:
        c = (b - a)/2 + a
    else:
        print("error")
        break
    if f(c)*f(a) < 0:
        b = c
    else:
        a = c

print(a, b)

def factorial(n):
    if n == 1:
        r = 1
    else:
        r = n*factorial(n-1)
    return r

print(factorial(3))
import numpy as np
punti = []
n = 10

for i in range(n):
    a = np.random.rand()
    b = np.random.rand()
    if np.sqrt(a**2 + b**2) < 1:
        punti.append((a, b))

print(punti)
print(len(punti))

m = 0

for i in range(len(l)):
    m += l[i]
m /= len(l)

s = 0
for i in range(len(l)):
    s += (l[i] - m)**2
s = np.sqrt(s / (len(l) - (len(l) - 1)))

print(m, s)

n = 100
x = 2

s=0
for i in range(n+1):
    s += x**i

print(s)
