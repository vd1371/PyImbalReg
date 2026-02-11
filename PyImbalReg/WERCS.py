# Loading dependencies
import pandas as pd
from .DataHandler import DataHandler

class WERCS(DataHandler):

    def __init__(self, **params):
        """WERCS: Weighted Relevance-based Combination Strategy.

        Ref: Branco et al., Neurocomputing 343, pp.76-99, 2019.

        Args:
            df: Data as pandas DataFrame.
            y_col_name: The name of the Y column header.
            rel_func: The relevance function.
            threshold: Threshold for rare vs normal.
            u_percentage: Fraction of (low-relevance) samples to keep when undersampling.
            o_percentage: Oversampling factor for high-relevance samples.
        """
        super().__init__(**params)

    def get(self):
        """Return the combined DataFrame (original + oversampled + undersampled)."""
        oversample_df = self.df.sample(
            frac=self.o_percentage - 1,
            replace=True,
            weights=self.Y_utility,
            random_state=self.random_state,
        )
        oversample_df.index = [
            f"OverSampled-{i}-{x}" for i, x in enumerate(oversample_df.index)
        ]
        undersample_df = self.df.sample(
            frac=1 - self.u_percentage,
            replace=True,
            weights=1 - self.Y_utility,
            random_state=self.random_state,
        )
        undersample_df.index = [
            f"UnderSampled-{i}-{x}" for i, x in enumerate(undersample_df.index)
        ]
        return pd.concat([self.df, oversample_df, undersample_df])