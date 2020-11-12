'''
This module is designed to oversample the rare samples.Ref:
Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
Pre-processing approaches for imbalanced distributions in regression.
Neurocomputing, 343, pp.76-99.
'''
import pandas as pd

from .DataHandler import DataHandler


class RandomOversampling(DataHandler):

	def __init__(self,
					df = pd.DataFrame(),         # The data as a pandas dataframe
					y_col = None,				 # The name of the Y column header
					rel_func = None,			 # The relevance function
					threshold = None,			 # Thereshol to dertermine the normal and reare samples
					o_percentage = 2			 # The oversampling percentage. This fraction - 1 will be added
					):
		super().__init__(df, y_col, rel_func, threshold)

		if self._is_o_percentage_correct(o_percentage):
			self.o_percentage = o_percentage

	def get(self):

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