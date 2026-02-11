# Loading dependencies
import pandas as pd
from .DataHandler import DataHandler

class RandomUndersampling(DataHandler):

    def __init__(self, **params):
        """Random undersampling of normal samples.

        Ref: Branco et al., Neurocomputing 343, pp.76-99, 2019.

        Args:
            df: Data as pandas DataFrame.
            y_col_name: The name of the Y column header.
            rel_func: The relevance function.
            threshold: Threshold to determine the normal and rare samples.
            u_percentage: Fraction of normal samples to keep (1 - u_percentage removed).
        """
        super().__init__(**params)

    def get(self):
        """Return the undersampled DataFrame."""
        undersampled_bins = []
        for normal_indices in self.normal_bins_indices:
            df = self.df.loc[normal_indices, :]
            undersampled_bins.append(
                df.sample(
                    frac=1 - self.u_percentage,
                    random_state=self.random_state,
                )
            )
        rare_bins = [
            self.df.loc[rare_indices, :]
            for rare_indices in self.rare_bins_indices
        ]
        return pd.concat(undersampled_bins + rare_bins)