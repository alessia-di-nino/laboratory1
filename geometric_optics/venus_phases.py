import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

file_path = "./geometric_optics/Phases_Venus.jpg"
plt.figure("fasi")
img = matplotlib.image.imread(file_path)
plt.imshow(img)
plt.show()
