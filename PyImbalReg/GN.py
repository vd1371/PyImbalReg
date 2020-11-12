'''
This module is designed to undersample a part of the normal cases
and add gaussian noise to the rare samples

Ref: 
Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
Pre-processing approaches for imbalanced distributions in regression.
Neurocomputing, 343, pp.76-99. 
'''
import numpy as np
import pandas as pd

from .DataHandler import DataHandler
from .RU import RandomUndersampling


class GaussianNoise(DataHandler):

	def __init__(self,
					df = pd.DataFrame(),         # The data as a pandas dataframe
					y_col = None,				 # The name of the Y column header
					rel_func = None,			 # The relevance function
					threshold = None,			 # Thereshol to dertermine the normal and reare samples
					u_percentage = 0.5,			 # The undersampling percentage. This fraction will be removed
					o_percentage = 0.5,			 # The oversampling percentage. (This fraction - 1) will be added
					perm_amp = 0.1,				 # The permutation amplitude
					categorical_columns = None	 # categorical columns will be used for generating new samples 
					):
		super().__init__(df, y_col, rel_func, thresholds)

		if self._is_u_percentage_correct(u_percentage):
			self.u_percentage = u_percentage

		if self._is_o_percentage_correct(o_percentage):
			self.o_percentage = o_percentage

		if self._is_perm_amp_correct(perm_amp):
			self.perm_amp = perm_amp

		# Finding the categorical columns
		if categorical_columns is None:
			categorical_columns = self.get_categorical_cols(df)
		
		self.categorical_columns = categorical_columns

	def get(self):

		# Undersampling the normal samples
		# Other parameters such as df, y_col, and threshold will be...
		# .. the same as the same parent of GaussianNoise. The prent is DataHandler
		ru = RandomUndersampling(u_percentage = self.u_percentage)
		undersample_df = ru.get()
		oversample_df = self._oversample_with_GN()

		# Concatenating the undersample normal samples and rare samples
		df = pd.concat([undersample_df, oversample_df])

		return df

	def _oversample_with_GN(self):

		# Over sampling the normal cases
		oversampled_bins = []

		for df in self.rare_bins:
			new_df = self._get_new_noisy_points(self.df, self.categorical_columns, self.o_percentage, self.perm_amp)
			oversampled_bins.append(new_df)

		return pd.concat(oversampled_bins)

	@staticmethod
	def _get_new_noisy_points(df, categorical_columns, o_percentage, perm_amp):

		# calculating the mean, std of the continuous variables
		# and finding the frequency of categorical variables
		info_dict = {}

		# CReating a new dataframe to pass as output
		new_df = pd.DataFrame(columns = df.columns)

		# Finding number of samples to be added
		n = int((o_percentage-1) * len(df))

		for col in df.columns:

			if col in categorical_columns:
				# Find the frequency of each cols
				# Value counts is a pd.Series. The index is the categories 
				# And the values are the probability of occurrence
				weights = df[col].value_counts(normalize = True)
				info_dict[col] = weights

				new_df[col] = np.random.choice(weights.index.tolist(),
												size = n,
												replace = True,
												p = weights.values.tolist())

			else:
				# Find the mean and std of the columns
				MEAN, STD = 0, 1
				info_dict[col] = [df[col].mean(), df[col].std()]

				# Oversampling values
				oversampled_values = np.random.choice(df[col].values, size = n)

				# Getting the noise to be added to the values
				noise = np.random.normal(loc = 0,
										scale = info_dict[col][STD] * perm_amp,
										size = n)
				# Adding the noise and values to the new_df
				new_df[col] = oversampled_values + noise

		# Renaming the indices for further references
		new_df.index = [ f"GN-{i}-{x}" for i, x in enumerate(new_df.index)]

		return new_df



