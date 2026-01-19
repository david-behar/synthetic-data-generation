"""CTGAN based balancer."""

from __future__ import annotations

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from .base import SyntheticBalancer


class CTGANBalancer(SyntheticBalancer):
    """CTGAN implementation for high-fidelity minority sample generation."""

    def __init__(self, target_col: str, strategy: str | float = 'auto', epochs: int = 100, verbose: bool = True) -> None:
        super().__init__(target_col=target_col, strategy=strategy)
        self.epochs = epochs
        self.verbose = verbose

    def _resample(self, df: pd.DataFrame):
        counts = df[self.target_col].value_counts()
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        target_count = counts[majority_class]
        current_minority_count = counts[minority_class]
        missing_rows = target_count - current_minority_count
        if missing_rows <= 0:
            print('Classes already balanced.')
            return df.copy(), pd.Series(False, index=df.index)
        print(f"Training CTGAN to generate {missing_rows} new rows for class '{minority_class}'...")
        df_minority = df[df[self.target_col] == minority_class]
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_minority)
        synthesizer = CTGANSynthesizer(metadata, epochs=self.epochs, verbose=self.verbose)
        synthesizer.fit(df_minority)
        synthetic_data = synthesizer.sample(num_rows=missing_rows)
        synthetic_data[self.target_col] = minority_class
        combined = pd.concat([df, synthetic_data], ignore_index=True)
        mask = pd.Series([False] * len(df) + [True] * len(synthetic_data), index=combined.index)
        return combined, mask
