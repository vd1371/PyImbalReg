# Loading dependencies
import numpy as np
import pandas as pd
from .DataHandler import DataHandler
from .GN import GaussianNoise

class GNHF(DataHandler):

    def __init__(self, **params):
        """Gaussian Noise with Histogram-based Frequencies.

        Undersamples/oversamples bins so each bin has count ~ mean_freq;
        uses Gaussian noise for oversampling.

        Args:
            df: Data as pandas DataFrame.
            y_col_name: The name of the Y column header.
            threshold: Threshold for binning (if used).
            perm_amp: Permutation amplitude for noise.
            categorical_columns: Columns treated as categorical.
            bins: Number of bins for the target histogram.
        """
        if params.pop("rel_func", None) is not None:
            raise ValueError("GNHF does not use rel_func; pass None.")
        super().__init__(**params)

    def get(self):
        """Return the resampled DataFrame (histogram-balanced with GN oversampling)."""
        freqs, edges = np.histogram(
            self.df.loc[:, self.y_col_name], bins=self.bins
        )
        if any(val <= 1 for val in freqs):
            raise ValueError(
                "A bin with 1 or 0 samples was found. "
                "Consider changing the number of bins."
            )

        mean_freq = np.mean(freqs)
        holder = []

        for freq, left_edge, right_edge in zip(freqs, edges[:-1], edges[1:]):
            bin_df = self.df[
                (self.df.loc[:, self.y_col_name] >= left_edge)
                & (self.df.loc[:, self.y_col_name] <= right_edge)
            ]
            ratio = mean_freq / freq
            if ratio < 1:
                new_df = bin_df.sample(frac=ratio)
                holder.append(new_df)
            else:
                new_df = GaussianNoise._get_new_noisy_points(
                    bin_df,
                    self.categorical_columns,
                    ratio,
                    self.perm_amp,
                )
                holder += [new_df, bin_df]

        return pd.concat(holder)