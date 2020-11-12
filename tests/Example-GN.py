# A simple example of addin gaussian noise 

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from seaborn import load_dataset
import pandas as pd

import PyImbalReg as pir


data = load_dataset('dots')

# Plotting the histogram of target values before oversampling
plt.hist(data.iloc[:, -1].values)
plt.title('Before')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()


def default_relevance_function(x):
	return 1 - norm_dist.pdf(x, loc = average, scale = std) / \
			norm_dist.pdf(average, loc = average, scale = std)
	return 0

gn = pir.GaussianNoise(data,
						rel_func = default_relevance_function,
						threshold = 0.7,
						o_percentage = 10,
						u_percentage = 0.3,
						perm_amp = 0.1)

new_data = gn.get()

# Plotting the histogram of target values after oversampling
plt.hist(new_data.iloc[:, -1].values)
plt.title('After')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()


