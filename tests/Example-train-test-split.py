# A simple example of addin gaussian noise

import matplotlib.pyplot as plt

from seaborn import load_dataset

import PyImbalRegTemp as pir

def evaluate(data, train, test, bins, method):
	# Creating a function to evaluate the splitting

	print (f" ------- With {method} ---------- ")
	print (f"Mean: Data: {data.iloc[:, -1].mean():.2f} " \
					f"train_set {train_set.iloc[:, -1].mean():.2f} " \
					f"test_set {test_set.iloc[:, -1].mean():.2f}")

	print (f"STD: Data: {data.iloc[:, -1].std():.2f} " \
				f"train_set {train_set.iloc[:, -1].std():.2f} " \
				f"test_set {test_set.iloc[:, -1].std():.2f}" )
	print (" ------------------------------------")

	fig, axs = plt.subplots(3)
	axs[0].hist(data.iloc[:, -1], bins = bins)
	axs[1].hist(train_set.iloc[:, -1], bins = bins)
	axs[2].hist(test_set.iloc[:, -1], bins = bins)
	plt.show()
	del fig, axs


# Loading the data
data = load_dataset('dots')

bins = 5

# Using PyImbalReg to train_test_split
train_set, test_set = pir.train_test_split(df = data,
							test_size = 0.2,
							bins = bins,
							random_state = 165)
evaluate (data, train_set, test_set, bins, "PyImbalReg")

# Using sklearn train_test_split
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data,
										random_state = 244)
evaluate (data, train_set, test_set, bins, "sklearn")

print ("Based on the results, the mean of the test set and train_set in sklearn "
		"are different from that of the data. But it's not the case in the PyImbalReg method")