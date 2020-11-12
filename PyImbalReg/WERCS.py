'''

This module is designed to apply WERCS algorithm for imbalanced regression.Ref:
Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
Pre-processing approaches for imbalanced distributions in regression.
Neurocomputing, 343, pp.76-99.
'''
import pandas as pd

from .DataHandler import DataHandler


class WERCS(DataHandler):

	def __init__(self,
					df = pd.DataFrame(),         # The data as a pandas dataframe
					y_col = None,				 # The name of the Y column header
					rel_func = None,			 # The relevance function
					u_percentage = 0.5,			 # The undersampling percentage. This fraction will be removed
					o_percentage = 2			 # The oversampling percentage. This fraction - 1 will be added
					):
		super().__init__(df, y_col, rel_func)

		if self._is_u_percentage_correct(u_percentage):
			self.u_percentage = u_percentage

		if self._is_o_percentage_correct(o_percentage):
			self.o_percentage = o_percentage

	def get(self):

		# Oversampling with relevance_function values
		oversample_df = self.df.sample(frac = self.o_percentage - 1, replace = True, weights = self.Y_utility)
		oversample_df.index = [ f"OverSampled-{i}-{x}" for i, x in enumerate(oversample_df.index)]

		# Undersampling with relevance_function values
		undersample_df = self.df.sample(frac = 1 - self.u_percentage, replace = True, weights = 1 - self.Y_utility)
		undersample_df.index = [ f"UnderSampled-{i}-{x}" for i, x in enumerate(undersample_df.index)]
		return pd.concat([self.df, oversample_df, undersample_df])