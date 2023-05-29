M = 4.44
sigma_M = 0.07

k = 0.0179
sigma_k = 0.0001

g = M*k
sigma_g = np.sqrt(k*k*sigma_M*sigma_M + M*M*sigma_k*sigma_k)

print(g, sigma_g)
