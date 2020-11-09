'''
This object is developed for handling the data before conduting any analysis on it
These processes include but not limited to checking the Nan files, data type, etc.
'''

import pandas as pd
import numpy as np
from scipy.stats import norm

class data_handler:

	def __init__(self, 
				df = pd.DataFrame(),
				y_col = None,
				rel_func = None,
				threshold = None):

		# The current version of PyImbalReg is designed to work only with pandas dataframes
		if not isinstance(df, pd.DataFrame):
			raise TypeError ("The current version of PyImbalReg can only work on pandas dataframes.")

		# The data must not contain any Nan values
		if df.isnull().values.any():
			raise ValueError ("The dataframe consists NaN values. Please consider removing them.")

		# Getting the Y column
		# If y is none, then the last column is considered as Y
		if y_col is None:
			self.Y = df.iloc[:, -1]

		# y should be either None or string
		elif not isinstance(y_col, str):
			raise TypeError ("y must be either None or a string")

		# if y is not one of the data columns
		elif not y_col in df.columns.values:
			raise ValueError ("y must be a column name, but it's not")

		# Then it's a column of the data
		else:
			self.Y = df.loc[:, y_col]

			# Rearrangin the dataframe so the last column be Y
			if df.columns.values[-1] != y_col:
				cols = df.columns.tolist()
				idx = cols.index(y_col)
				cols = cols[:idx] + cols[idx+1:] + [cols[idx]]
				df = df[cols]

		self.df = df

		'''
		Set the relevance function and threshold
		Relevenace function is a relevance/utility function that maps the Y to [0, 1]
		Values of u(Y) > threshold are considered as rare samples

		Ref: 
		Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
		Pre-processing approaches for imbalanced distributions in regression.
		Neurocomputing, 343, pp.76-99.
		'''

		self.set_relevance_function(rel_func, threshold)


	def set_relevance_function(self, rel_func, threshold):

		# The default behaviour
		if rel_func is None:
			average, std = self.Y.mean(), self.Y.std()

			# Default relevance function is based on probability distribution function ...
			# ... of normal distribution
			def default_rel_func(x, average = average, std = std, norm_dist = norm):
				return 1 - norm_dist.pdf(x, loc = average, scale = std) / \
							 norm_dist.pdf(average, loc = average, scale = std)

			self.rel_func = default_rel_func

		# Check if the rel_fun is a 
		elif not callable(rel_func):
			raise TypeError ("The rel_func is expected to be a fuction, but it's not")

		# Set the relevance function
		else:
			self.rel_func = rel_func

		if any(self.rel_func(self.Y) > 1) or any(self.rel_func(self.Y) < 0):
			raise ValueError ("It is expected that the relevance function returns\
								values between [0, 1]. But it doesn't. Please re-define your function")

		# Check if the threshold is a float
		if not isinstance(threshold, (int, float)):
			raise ValueError ("The threshold must be float")
		
		# Check if the threshold is between 0 and 1
		elif not (threshold > 0 and threshold < 1):
			raise ValueError ("The threshold must be between [0,1]. But it's not.")

		self.threshold = threshold






