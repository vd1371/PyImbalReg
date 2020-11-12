# A simple example of addin gaussian noise 

import matplotlib.pyplot as plt
from seaborn import load_dataset
import PyImbalReg as pir


data = load_dataset('dots')

# Plotting the histogram of target values before oversampling
plt.hist(data.iloc[:, -1].values)
plt.title('Before')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

# Required values to define the user defined relevance funtion
average, std = data.iloc[:, -1].mean(), data.iloc[:, -1].std()
from scipy.stats import norm
def relevance_function(x, average = average, std = std):
	# This function is used as the 1 - normalized values of normal distribution
	# When no relevance function is defined, this function will be used
	return 1 - norm.pdf(x, loc = average, scale = std) / \
			norm.pdf(average, loc = average, scale = std)

# Oversampling and undersampling using WERCS
wercs = pir.WERCS(df = data,
					y_col = 'firing_rate',
					rel_func = relevance_function,
					u_percentage = 0.5,
					o_percentage = 2)

new_data = wercs.get()

# Plotting the histogram of target values after oversampling
plt.hist(new_data.iloc[:, -1].values)
plt.title('After')
plt.xlabel("Values")
plt.ylabel("Frequency")
plt.show()

