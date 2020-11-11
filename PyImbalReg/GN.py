'''
This module is designed to undersample a part of the normal cases
and add gaussian noise to the rare samples

Ref: 
Branco, P., Torgo, L. and Ribeiro, R.P., 2019.
Pre-processing approaches for imbalanced distributions in regression.
Neurocomputing, 343, pp.76-99. 
'''
import pandas as pd
import warnings

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
		super().__init__(df, y_col, rel_func, threshold)

		if self._is_u_percentage_correct(u_percentage):
			self.u_percentage = u_percentage

		if self._is_o_percentage_correct(o_percentage):
			self.o_percentage = o_percentage

		if self._is_perm_amp_correct(perm_amp):
			self.perm_amp = perm_amp

		# Finding the categorical columns
		if categorical_columns is None:
			warnings.warn("The categorical_columns is not define by you.\
							I will try to find the categorical columns using heuristic methods. \
							There is a small chance it fails. Consider passing the categorical_columns.", UserWarning)
		
			nominal_dtypes = ['object', 'bool', 'datetime64']
			
			categorical_columns = []
			for col in self.df.columns:

				# Adding the string, boolean, and datetimes columns
				if self.df[col].dtypes in nominal_dtypes:
					categorical_columns.append(col)

				# Checking the integer columns
				elif pd.api.types.is_integer_dtype(self.df[col].dtypes):
					
					# A heuristic method to check if the column in categorical
					if df[col].nunique() / df[col].count() < 0.05:
						categorical_columns.append(col)

		self.categorical_columns = categorical_columns

	def get(self):

		# Undersampling the normal samples
		ru = RandomUndersampling(df = self.df,
									y_col = self.y_col,
									rel_func = self.rel_func,
									threshold = self.threshold,
									u_percentage = self.u_percentage)

		undersample_df = ru.get()

		print (undersample_df)

		# Finding number of samples to be added
		n = int((self.o_percentage-1) * len(self.df_rare))

		# 