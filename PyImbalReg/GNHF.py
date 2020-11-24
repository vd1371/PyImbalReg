# Loading dependencies
import numpy as np
import pandas as pd
from .DataHandler import DataHandler
from .GN import GaussianNoise

class GNHF(DataHandler):

	def __init__(self, **params):
		'''Contructor params:

		This module is designed to undersample a part of the normal cases
		and add gaussian noise to the rare samples based on the histogram frequencies

		The oversampling and under sampling ratios are based on histogram frequency
		Ratio: mean_frequency/frequency

		df: Data as pandas dataframe
		y_col: The name of the Y column header
		threshold: Thereshold to dertermine the normal and reare samples
		perm_amp: The permutation amplitude
		categorical_columns: categorical columns will be used for generating new samples
		bins: number of bins for creating histogram
		'''
		super().__init__(**params)

	def get(self):
		"""getting the output"""

		freqs, edges = np.histogram(self.Y, bins = self.bins)
		mean_freq = np.mean(freqs)

		holder = []
		for freq, left_ege, right_edge in zip(freqs, edges[:-1], edges[1:]):

			bin_df = self.df[(self.Y >= left_ege) & (self.Y <= right_edge)]
			if len(bin_df) > 0:
				ratio = mean_freq / freq
				# Undersampling given the ratio
				if ratio < 1:
					new_df = bin_df.sample(frac = ratio)
					holder += [new_df]
				# Oversampling with GN given the ratio
				else:
					new_df = GaussianNoise._get_new_noisy_points(
													bin_df,
													self.categorical_columns,
													ratio,
													self.perm_amp)
					holder += [new_df, bin_df]

		df = pd.concat(holder)

		return df