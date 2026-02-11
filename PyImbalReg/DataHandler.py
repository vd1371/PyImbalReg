# Loading dependencies
import numpy as np
import pandas as pd
import warnings
from scipy.stats import norm


class DataHandler:

    def __init__(self, **params):
        """Build the base for other methods built upon this.

        This object handles the data before conducting any analysis.
        Processes include checking for NaN, data types, etc.

        Args:
            df: The data as a pandas DataFrame.
            y_col_name: The name of the Y column header.
            rel_func: The relevance function.
            threshold: Threshold to determine the normal and rare samples.
            should_log_transform: Useful when there is a huge difference
                between the order of the target values.
        """
        df = params.pop("df", None)
        y_col_name = params.pop("y_col_name", None)
        rel_func = params.pop("rel_func", None)
        threshold = params.pop("threshold", 0.9)
        o_percentage = params.pop("o_percentage", 2)
        u_percentage = params.pop("u_percentage", 0.2)
        perm_amp = params.pop("perm_amp", 0.1)
        categorical_columns = params.pop("categorical_columns", None)
        bins = params.pop("bins", 10)
        should_log_transform = params.pop("should_log_transform", False)
        random_state = params.pop("random_state", None)
        self.should_sort = params.pop("should_sort", True)


        self.random_state = random_state
        np.random.seed(random_state)

        # The current version of PyImbalReg is designed to work only with pandas dataframes
        if not isinstance(df, pd.DataFrame):
            raise TypeError("The current version of PyImbalReg can "\
                                "only work on pandas dataframes.")

        # The data must not contain any Nan values
        if df.isnull().values.any():
            raise ValueError("The dataframe consists NaN values. "\
                                    "Please consider removing them.")

        # Getting the Y column
        if y_col_name is None:
            y_col_name = df.columns.values[-1]

        # y should be either None or string
        elif not isinstance(y_col_name, str):
            raise TypeError("y must be either None or a string")

        # if y is not one of the data columns
        elif y_col_name not in df.columns.values:
            raise ValueError("y must be a column name, but it's not")

        self.y_col_name = y_col_name

        # Rearrange the dataframe so the last column is Y
        if df.columns.values[-1] != y_col_name:
            cols = df.columns.tolist()
            idx = cols.index(y_col_name)
            cols = cols[:idx] + cols[idx+1:] + [cols[idx]]
            df = df[cols]

        self.df = df

        # Relevance function maps Y to [0, 1]. Values u(Y) > threshold are rare.
        # Ref: Branco et al., Neurocomputing 343, pp.76-99, 2019.

        self.o_percentage = self._is_o_percentage_correct(o_percentage)
        self.u_percentage = self._is_u_percentage_correct(u_percentage)
        self.perm_amp = self._is_perm_amp_correct(perm_amp)
        self.bins = self._is_bins_correct(bins)

        # Finding the categorical columns
        if categorical_columns is None:
            categorical_columns = self.get_categorical_cols(df)
        self.categorical_columns = categorical_columns

        # Setting the relevance function, normal bins, rare bins, ...
        # Some algorithms do not need rel_func.
        if rel_func is not None:
            # Setting the relevance function and threshold
            self.set_relevance_function(rel_func, threshold)

            # Finding the rare and normal values
            self.find_normal_rare_values()

    # Set the undersampling percentage
    def set_u_percentage(self, u_percentage):
        self.u_percentage = self._is_u_percentage_correct(u_percentage)

    # Set the oversampling percentage
    def set_o_percentage(self, o_percentage):
        self.o_percentage = self._is_o_percentage_correct(o_percentage)

    # Assigning the relevance function and the threshold
    def set_relevance_function(self, rel_func, threshold):

        # The default behaviour
        if rel_func == 'default' or rel_func is None:
            average, std = self.df.loc[:, self.y_col_name].mean(), self.df.loc[:, self.y_col_name].std()

            # Default relevance function is based on probability distribution function ...
            # ... of normal distribution
            def default_rel_func(x, average = average, std = std):
                return 1 - norm.pdf(x, loc = average, scale = std) / \
                             norm.pdf(average, loc = average, scale = std)

            self.rel_func = default_rel_func

        # Check if the rel_fun is a function
        elif not callable(rel_func):
            raise TypeError("The rel_func is expected to be a function, but it's not")

        # Set the relevance function
        else:
            self.rel_func = rel_func

        # Check if the threshold is a float
        if not isinstance(threshold, (float)):
            raise ValueError("The threshold must be float")
        # Check if the threshold is between 0 and 1
        elif not (threshold > 0 and threshold < 1):
            raise ValueError("The threshold must be between [0,1]. But it's not.")

        self.threshold = threshold

    # Finding the relevance value of the Y
    def find_normal_rare_values(self):

        # Finding bins with the normal Y and rare Y
        self.rare_bins_indices, self.normal_bins_indices = [], []

        if self.should_sort:
            # Sorting the values of df
            self.df.sort_values(self.df.columns[-1], inplace = True)
        
        # Resetting index for correct splitting in the following steps
        self.df.reset_index(drop = False, inplace = True)

        # Finding the relevance value of the Y
        self.df['utility'] = self.df.loc[:, self.y_col_name].apply(self.rel_func)

        if any(self.df.loc[:, 'utility'] > 1) or any(self.df.loc[:, 'utility'] < 0):
            raise ValueError("It is expected that the relevance function returns\
                                values between [0, 1]. But it doesn't. Please re-define your relevance function")

        # Assuming all samples are normal at the beginning: Normal = 1
        # And adding the 'R/N' columns for next steps
        self.df.loc[:, 'R/N'] = 1

        # Finding the rare values
        self.df.loc[self.df['utility'] >= self.threshold, 'R/N'] = -1

        # Shifting the R/N and multiplying it with the original one
        # Those places where this value is equal to -1 are the boundaries of bins
        changing_points = self.df.loc[:, 'R/N'] * self.df.loc[:, 'R/N'].shift(periods = 1)
        
        # Finding the changing points indices (left side of each bin)
        changing_points_idx = self.df[changing_points == -1].index.tolist()

        # Add 0 and len(df) for indexing
        changing_points_idx = [0] + changing_points_idx + [len(self.df)]
        for left, right in zip(changing_points_idx[:-1], changing_points_idx[1:]):
            
            # Selecting the bins' indices
            selected_bin = self.df.loc[:, 'index'].iloc[left:right].tolist()
            if self.df.loc[left, 'R/N'] == 1:
                self.normal_bins_indices.append(selected_bin)
            else:
                self.rare_bins_indices.append(selected_bin)

        # Setting the original index as the index one more time
        self.df.set_index('index', drop = True, inplace = True)

        # Slicing the utility for future use
        self.Y_utility = self.df.loc[:, 'utility'].copy()

        # dropping the newly created columns
        self.df.drop(columns = ['R/N', 'utility'], inplace = True)


    # Checking if the o_percentage is correct
    @staticmethod
    def _is_o_percentage_correct(o_percentage):
        # o_percentage : Oversampling percentage

        # Check if the o_percentage is a float
        if not isinstance(o_percentage, (int, float)):
            raise ValueError("The o_percentage must be float")
        # Check if the o_percentage is between 0 and 1
        elif not (o_percentage > 1):
            raise ValueError("The o_percentage must be bigger than 1. But it's not.")

        return o_percentage

    # Checking if the u_percentage is correct
    @staticmethod
    def _is_u_percentage_correct(u_percentage):
        # u_percentage : Oversampling percentage

        # Check if the u_percentage is a float
        if not isinstance(u_percentage, (float)):
            raise ValueError("The u_percentage must be float")
        # Check if the u_percentage is between 0 and 1
        elif not (u_percentage > 0 and u_percentage < 1):
            raise ValueError("The u_percentage must be between [0,1]. But it's not.")

        return u_percentage

    # Checking if the permutation amplitude is a float
    @staticmethod
    def _is_perm_amp_correct(perm_amp):
        # perm_amp: permutation amplitude
        if not isinstance(perm_amp, (int, float)):
            raise ValueError("The perm_amp must be float")

        return perm_amp

    # Checking if the bins is an integer
    @staticmethod
    def _is_bins_correct(bins):
        # bins: number of bins for making histogram
        if not isinstance(bins, int):
            raise ValueError("The bins must be integer")

        return bins

    # Getting the categorical columns of a dataframe
    @staticmethod
    def get_categorical_cols(df):
        # This method is called when the categorical columns are not passed by the user

        warning_message = "\n\n---------------------------------------\n" +\
                        "The categorical_columns is not defined by you.\n" +\
                        "I will try to find the categorical columns using heuristic methods.\n" +\
                        "There is a small chance it fails. Consider passing the categorical_columns. \n" +\
                        "---------------------------------------\n"
        warnings.warn(warning_message, UserWarning, stacklevel=2)

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






