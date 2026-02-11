# Loading dependencies
import pandas as pd
from .DataHandler import DataHandler

class RandomOversampling(DataHandler):

    def __init__(self, **params):
        """Random oversampling of rare samples.

        Ref: Branco et al., Neurocomputing 343, pp.76-99, 2019.

        Args:
            df: Data as pandas DataFrame.
            y_col_name: The name of the Y column header.
            rel_func: The relevance function.
            threshold: Threshold to determine the normal and rare samples.
            o_percentage: Oversampling factor; (o_percentage - 1) * n_rare samples added.
        """
        super().__init__(**params)

    def get(self):
        """Return the oversampled DataFrame."""
        oversampled_bins = []

        for rare_indices in self.rare_bins_indices:
            df = self.df.loc[rare_indices, :]
            oversample_df = df.sample(
                frac=self.o_percentage - 1,
                replace=True,
                random_state=self.random_state,
            )
            oversample_df.index = [
                f"OverSampled-{i}-{x}" for i, x in enumerate(oversample_df.index)
            ]
            oversampled_bins += [oversample_df, df]

        normal_bins = [
            self.df.loc[normal_indices, :]
            for normal_indices in self.normal_bins_indices
        ]
        return pd.concat(oversampled_bins + normal_bins)