# Loading dependencies
import pandas as pd
from .DataHandler import DataHandler

class RandomOversampling(DataHandler):

	def __init__(self, **params):
		'''Contructor params:

		This module is designed to oversample the rare samples.Ref:
		Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
		Pre-processing approaches for imbalanced distributions in regression.
		Neurocomputing, 343, pp.76-99.

		df: Data as pandas dataframe
		y_col: The name of the Y column header
		rel_func: The relevance function
		threshold: Thereshold to dertermine the normal and reare samples
		o_percentage: The oversampling percentage. (This fraction - 1) will be added
		'''
		super().__init__(**params)

	def get(self):
		"""getting the output"""
		oversampled_bins = []

		for rare_indices in self.rare_bins_indices:

			df = self.df.loc[rare_indices, :]
			# Over sampling the rare cases
			oversample_df = df.sample(frac = self.o_percentage - 1, 
									replace = True,
									random_state = self.random_state)
			oversample_df.index = [ f"OverSampled-{i}-{x}" for i, x in enumerate(oversample_df.index)]

			# Adding the df and the oversampled df to the oversampled bins
			oversampled_bins += [oversample_df , df]

		# Getting the normal bins
		normal_bins = []
		for normal_indices in self.normal_bins_indices:
			normal_bins.append(self.df.loc[normal_indices, :])

		# Concatenating the undersample normal samples and rare samples
		df = pd.concat(oversampled_bins + normal_bins)

		return df