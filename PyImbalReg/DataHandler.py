'''
This object is developed for handling the data before conduting any analysis on it
These processes include but not limited to checking the Nan files, data type, etc.
'''

import pandas as pd
import numpy as np
import warnings
from scipy.stats import norm

class DataHandler:

	instance = None

	def __init__(self, 
				df = pd.DataFrame(),         # The data as a pandas dataframe
				y_col = None,				 # The name of the Y column header
				rel_func = None,			 # The relevance function
				threshold = None,			 # Thereshol to dertermine the normal and reare samples
				**kwargs):

		# Not to instantiate the DataHandler more than once
		if DataHandler.instance is None:

			DataHandler.instance = True

			# The current version of PyImbalReg is designed to work only with pandas dataframes
			if not isinstance(df, pd.DataFrame):
				raise TypeError ("The current version of PyImbalReg can only work on pandas dataframes.")

			# The data must not contain any Nan values
			if df.isnull().values.any():
				raise ValueError ("The dataframe consists NaN values. Please consider removing them.")

			# Getting the Y column
			# If y is none, then the last column is considered as Y
			if y_col is None:
				DataHandler.Y = df.iloc[:, -1]

			# y should be either None or string
			elif not isinstance(y_col, str):
				raise TypeError ("y must be either None or a string")

			# if y is not one of the data columns
			elif not y_col in df.columns.values:
				raise ValueError ("y must be a column name, but it's not")

			# Then it's a column of the data
			else:
				DataHandler.Y = df.loc[:, y_col]

				# Rearrangin the dataframe so the last column be Y
				if df.columns.values[-1] != y_col:
					cols = df.columns.tolist()
					idx = cols.index(y_col)
					cols = cols[:idx] + cols[idx+1:] + [cols[idx]]
					df = df[cols]

			DataHandler.df = df
			DataHandler.y_col = y_col

			'''
			Set the relevance function and threshold
			Relevenace function is a relevance/utility function that maps the Y to [0, 1]
			Values of u(Y) > threshold are considered as rare samples

			Ref: 
			Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
			Pre-processing approaches for imbalanced distributions in regression.
			Neurocomputing, 343, pp.76-99.
			'''
			DataHandler.set_relevance_function(rel_func, threshold)

	# Assigning the relevance function and the threshold
	def set_relevance_function(rel_func, threshold):

		# The default behaviour
		if rel_func is None:
			average, std = DataHandler.Y.mean(), DataHandler.Y.std()

			# Default relevance function is based on probability distribution function ...
			# ... of normal distribution
			def default_rel_func(x, average = average, std = std, norm_dist = norm):
				return 1 - norm_dist.pdf(x, loc = average, scale = std) / \
							 norm_dist.pdf(average, loc = average, scale = std)

			DataHandler.rel_func = default_rel_func

		# Check if the rel_fun is a 
		elif not callable(rel_func):
			raise TypeError ("The rel_func is expected to be a fuction, but it's not")

		# Set the relevance function
		else:
			DataHandler.rel_func = rel_func

		if any(DataHandler.rel_func(DataHandler.Y) > 1) or any(DataHandler.rel_func(DataHandler.Y) < 0):
			raise ValueError ("It is expected that the relevance function returns\
								values between [0, 1]. But it doesn't. Please re-define your function")

		# Check if the threshold is a float
		if not isinstance(threshold, (float)):
			raise ValueError ("The threshold must be float")
		
		# Check if the threshold is between 0 and 1
		elif not (threshold > 0 and threshold < 1):
			raise ValueError ("The threshold must be between [0,1]. But it's not.")

		DataHandler.threshold = threshold

		# Finding the rare and normal values
		DataHandler.set_relevance_values()

	# Finding the relevance value of the Y
	def set_relevance_values():

		# Finding bins with the normal Y and rare Y
		DataHandler.rare_bins, DataHandler.normal_bins = [], []

		# Sorting the values of df
		DataHandler.df.sort_values(DataHandler.df.columns[-1], inplace = True)

		RARE, NORMAL = True, False
		bin_indices = []

		previous_status = DataHandler.rel_func(DataHandler.df.iloc[0, -1]) >= DataHandler.threshold
		start_idx = 0
		for i, idx in enumerate(DataHandler.df.index[1:]):

			# Check if it's a rare case or not
			current_status = DataHandler.rel_func(DataHandler.df.iloc[i, -1]) >= DataHandler.threshold

			if not current_status == previous_status or idx == DataHandler.df.index[-1]:
				temp_bin = DataHandler.df.iloc[start_idx:last_idx, :]
				if previous_status == RARE:
					DataHandler.rare_bins.append(temp_bin)
				else:
					DataHandler.normal_bins.append(temp_bin)

				start_idx = i

			last_idx = i
			previous_status = current_status

		# Finding the relevance value of the Y
		DataHandler.Y_utility = DataHandler.rel_func(DataHandler.Y)

	# Checcikng if the o_percentage is correct
	@staticmethod
	def _is_o_percentage_correct(o_percentage):
		# o_percentage : Oversampling percentage

		# Check if the o_percentage is a float
		if not isinstance(o_percentage, (int, float)):
			raise ValueError ("The o_percentage must be float")
		
		# Check if the o_percentage is between 0 and 1
		elif not (o_percentage > 1):
			raise ValueError ("The o_percentage must be bigger than 1. But it's not.")

		return True

	# Checcikng if the u_percentage is correct
	@staticmethod
	def _is_u_percentage_correct(u_percentage):
		# u_percentage : Oversampling percentage

		# Check if the u_percentage is a float
		if not isinstance(u_percentage, (float)):
			raise ValueError ("The u_percentage must be float")
		
		# Check if the u_percentage is between 0 and 1
		elif not (u_percentage > 0 and u_percentage < 1):
			raise ValueError ("The u_percentage must be between [0,1]. But it's not.")

		return True

	# Checking if the permutation amplitude is a float
	@staticmethod
	def _is_perm_amp_correct(perm_amp):
		# perm_amp: permutation amplitude
		if not isinstance(perm_amp, (int, float)):
			raise ValueError ("The perm_amp must be float")

		return True

	# Getting the categorical columns of a dataframe
	@staticmethod
	def get_categorical_cols(df):

		# This method is called when the categorical columns are not passed by the user

		warning_message = "\n\n---------------------------------------\n" +\
						"The categorical_columns is not define by you.\n" +\
						"I will try to find the categorical columns using heuristic methods.\n" +\
						"There is a small chance it fails. Consider passing the categorical_columns. \n" +\
						"---------------------------------------\n"
		warnings.warn(warning_message, UserWarning , stacklevel = 2)

		nominal_dtypes = ['object', 'bool', 'datetime64']

		categorical_columns = []
		for col in df.columns:

			# Adding the string, boolean, and datetimes columns
			if df[col].dtypes in nominal_dtypes:
				categorical_columns.append(col)

			# Checking the integer columns
			elif pd.api.types.is_integer_dtype(df[col].dtypes):
				
				# A heuristic method to check if the column in categorical
				if df[col].nunique() / df[col].count() < 0.05:
					categorical_columns.append(col)

		return categorical_columns







