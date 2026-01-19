"""Base classes for synthetic data balancers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Tuple
import pandas as pd


class SyntheticBalancer(ABC):
    """Common interface for class balancing strategies."""

    def __init__(
        self,
        target_col: str,
        strategy: str | float = 'auto',
        synthetic_flag_col: str = '_is_generated',
    ) -> None:
        self.target_col = target_col
        self.strategy = strategy
        self.synthetic_flag_col = synthetic_flag_col

    def fit_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run balancing strategy and report distributions."""
        print(f"--- Starting Balancing using {self.__class__.__name__} ---")
        self._log_distribution(df, label='Original')
        df_balanced, synthetic_mask = self._resample(df)
        synthetic_mask = self._coerce_mask(df_balanced, synthetic_mask)
        if self.synthetic_flag_col in df_balanced.columns:
            raise ValueError(
                f"Synthetic flag column '{self.synthetic_flag_col}' already exists in the input dataframe."
            )
        df_balanced = df_balanced.copy()
        df_balanced[self.synthetic_flag_col] = synthetic_mask.values
        self._log_distribution(df_balanced, label='New')
        return df_balanced

    def _log_distribution(self, df: pd.DataFrame, label: str) -> None:
        print(f"{label} Class Distribution:\n{df[self.target_col].value_counts()}")

    @abstractmethod
    def _resample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Iterable[bool]]:
        """Implementation-specific balancing logic.

        Returns a tuple of the balanced dataframe and a boolean-like iterable
        indicating which rows are synthetic (True) versus original (False).
        """
        raise NotImplementedError

    def _coerce_mask(self, df_balanced: pd.DataFrame, mask: Iterable[bool]) -> pd.Series:
        if isinstance(mask, pd.Series):
            mask_series = mask.copy()
        else:
            mask_series = pd.Series(mask)
        if len(mask_series) != len(df_balanced):
            raise ValueError(
                "Synthetic indicator mask must be the same length as the balanced dataframe."
            )
        if not mask_series.index.equals(df_balanced.index):
            mask_series = mask_series.reset_index(drop=True)
            mask_series.index = df_balanced.index
        return mask_series.astype(bool)
