# Loading dependencies
import numpy as np
import pandas as pd
from .DataHandler import DataHandler
from .RU import RandomUndersampling

class GaussianNoise(DataHandler):

    def __init__(self, **params):
        """Undersample normal cases and oversample rare cases with Gaussian noise.

        Ref: Branco et al., Neurocomputing 343, pp.76-99, 2019.

        Args:
            df: Data as pandas DataFrame.
            y_col_name: The name of the Y column header.
            rel_func: The relevance function.
            threshold: Threshold to determine the normal and rare samples.
            u_percentage: Fraction of normal samples to keep.
            o_percentage: Oversampling factor for rare samples.
            perm_amp: Permutation amplitude for added noise.
            categorical_columns: Columns treated as categorical for sampling.
        """
        super().__init__(**params)

    def get(self):
        """Return the resampled DataFrame (undersampled normal + GN oversampled rare)."""
        ru = RandomUndersampling(**self.__dict__)
        undersample_df = ru.get()
        oversample_df = self._oversample_with_GN()

        # Concatenating the undersample normal samples and rare samples
        df = pd.concat([undersample_df, oversample_df])

        return df

    def _oversample_with_GN(self):
        """Oversample rare bins by adding Gaussian noise."""
        oversampled_bins = []

        for rare_indices in self.rare_bins_indices:
            df = self.df.loc[rare_indices, :]
            new_df = self._get_new_noisy_points(df, self.categorical_columns, self.o_percentage, self.perm_amp)
            oversampled_bins += [df, new_df]

        return pd.concat(oversampled_bins)

    @staticmethod
    def _get_new_noisy_points(df, categorical_columns, o_percentage, perm_amp):
        """Generate new synthetic points by adding Gaussian noise.

        Args:
            df: Source DataFrame.
            categorical_columns: List of categorical column names.
            o_percentage: Oversampling factor.
            perm_amp: Noise scale (fraction of column std).

        Returns:
            New DataFrame with synthetic noisy samples.
        """
        new_df = pd.DataFrame(columns=df.columns)
        n = int((o_percentage - 1) * len(df))

        for col in df.columns:

            if col in categorical_columns:
                counts = df[col].value_counts(normalize=True)
                weights = counts.values  # already probabilities (sum=1)

                new_df[col] = np.random.choice(
                    counts.index.tolist(),
                    size=n,
                    replace=True,
                    p=weights,
                )

            else:
                oversampled_values = np.random.choice(
                    df[col].values, size=n, replace=True
                )
                std = df[col].std()
                noise = np.random.normal(
                    loc=0, scale=std * perm_amp, size=n
                )
                new_df[col] = oversampled_values + noise

        new_df.index = [f"GN-{i}-{x}" for i, x in enumerate(new_df.index)]

        return new_df




