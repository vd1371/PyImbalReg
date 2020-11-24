'''train_test_split

The idea is to develop atrain_test_split that randomly split the data
so the distribution of the test_set and the train set are similar. The train_test_split
provided by sklearn library does not consider the imbalanced problem in datasets

'''

import pandas as pd
import numpy as np

def train_test_split (df = pd.DataFrame(),
					test_size = None,
					bins = None,
					random_state = None
					):
	'''Splits a dataframe to train and test dataframes with similar distributions

	params:
	df: A data frame with the last column as target value
	test_size: A float between 0 and 1, indicates the portion of test set
	bins: Number of bins to make histogram with
	random_state: random_state to make sampling reproducable
	Returns: Two pandas dataframes: Train set and test set
	'''
	# The current version of PyImbalReg is designed to work only with pandas dataframes
	if not isinstance(df, pd.DataFrame):
		raise TypeError ("The current version of PyImbalReg can only work on pandas dataframes.")

	# Check if the test_size is a float
	if not isinstance(test_size, (float)):
		raise ValueError ("The test_size must be float")
	# Check if the test_size is between 0 and 1
	elif not (test_size > 0 and test_size < 1):
		raise ValueError ("The test_size must be between [0,1]. But it's not.")

	# Check if the random_state is an integer
	if not random_state is None and not isinstance(random_state, int):
		raise ValueError ("The random_state must be integer")

	# Check if the bins is an integer
	if not isinstance(bins, int):
		raise ValueError ("The bins must be integer")

	# Get the last column as the target value
	y = df.iloc[:, -1].values

	_, bin_edges = np.histogram(y)

	test_dfs_list = []
	train_dfs_list = []

	for left_edge, right_edge in zip (bin_edges[:-1], bin_edges[1:]):

		bin_df = df[(y > left_edge) & (y < right_edge)]

		test_df = bin_df.sample(frac = test_size, random_state = random_state)
		train_df = bin_df.drop(test_df.index)

		test_dfs_list.append(test_df)
		train_dfs_list.append(train_df)

	test_df = pd.concat(test_dfs_list)
	train_df = pd.concat(train_dfs_list)

	return train_df, test_df
