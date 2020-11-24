'''Randomoversampling

This module is designed to oversample the rare samples.Ref:
Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
Pre-processing approaches for imbalanced distributions in regression.
Neurocomputing, 343, pp.76-99.
'''
import pandas as pd

from .DataHandler import DataHandler


class RandomOversampling(DataHandler):

	def __init__(self, **params):
		'''Contructor params:

		df: Data as pandas dataframe
		y_col: The name of the Y column header
		rel_func: The relevance function
		threshold: Thereshold to dertermine the normal and reare samples
		o_percentage: The oversampling percentage. (This fraction - 1) will be added
		'''
		super().__init__(**params)

	def get(self):
		"""getting the output"""

		# Over sampling the normal cases
		oversampled_bins = []

		for df in self.rare_bins:

			# Over sampling the rare cases
			oversample_df = df.sample(frac = self.o_percentage - 1, replace = True)
			oversample_df.index = [ f"OverSampled-{i}-{x}" for i, x in enumerate(oversample_df.index)]

			# Adding the df and the oversampled df to the oversampled bins
			oversampled_bins += [oversample_df , df]

		# Concatenating the undersample normal samples and rare samples
		df = pd.concat(oversampled_bins + self.normal_bins)

		return df