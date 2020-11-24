# Loading dependencies
import pandas as pd
from .DataHandler import DataHandler

class RandomUndersampling(DataHandler):

	def __init__(self, **params):
		'''Contructor params:

		This module is designed to undersample the normal samples.
		Ref:
		Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
		Pre-processing approaches for imbalanced distributions in regression.
		Neurocomputing, 343, pp.76-99.

		df: Data as pandas dataframe
		y_col: The name of the Y column header
		rel_func: The relevance function
		threshold: Thereshold to dertermine the normal and reare samples
		u_percentage: The undersampling percentage. This fraction will be removed'''
		super().__init__(**params)

	def get(self):
		"""Getting the new data"""
		undersampled_bins = []

		for df in self.normal_bins:
			# Finding number of sample to be selected
			undersampled_bins.append(df.sample(frac = 1 - self.u_percentage))

		# Concatenating the undersample normal samples and rare samples
		df = pd.concat(undersampled_bins + self.rare_bins)

		return df