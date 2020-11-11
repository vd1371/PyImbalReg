'''
This module is designed to undersample the normal samples.

Ref: 
Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
Pre-processing approaches for imbalanced distributions in regression.
Neurocomputing, 343, pp.76-99.
'''
import pandas as pd

from .DataHandler import DataHandler


class RandomUndersampling(DataHandler):

	def __init__(self,
					df = pd.DataFrame(),         # The data as a pandas dataframe
					y_col = None,				 # The name of the Y column header
					rel_func = None,			 # The relevance function
					threshold = None,			 # Thereshol to dertermine the normal and reare samples
					u_percentage = 0.5			 # The undersampling percentage. This fraction will be removed
					):
		super().__init__(df, y_col, rel_func, threshold)

		if self._is_u_percentage_correct(u_percentage):
			self.u_percentage = u_percentage

	def get(self):

		# Under sampling the normal cases
		undersampled_bins = []

		for df in self.normal_bins:
			# Finding number of sample to be selected
			n = int((1 - self.u_percentage) * len(df))
			undersampled_bins.append(df.sample(n = n))

		# Concatenating the undersample normal samples and rare samples
		df = pd.concat(undersampled_bins + self.rare_bins)

		return df