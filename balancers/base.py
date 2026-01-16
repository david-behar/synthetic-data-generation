"""Base classes for synthetic data balancers."""

from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class SyntheticBalancer(ABC):
    """Common interface for class balancing strategies."""

    def __init__(self, target_col: str, strategy: str | float = 'auto') -> None:
        self.target_col = target_col
        self.strategy = strategy

    def fit_resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run balancing strategy and report distributions."""
        print(f"--- Starting Balancing using {self.__class__.__name__} ---")
        self._log_distribution(df, label='Original')
        df_balanced = self._resample(df)
        self._log_distribution(df_balanced, label='New')
        return df_balanced

    def _log_distribution(self, df: pd.DataFrame, label: str) -> None:
        print(f"{label} Class Distribution:\n{df[self.target_col].value_counts()}")

    @abstractmethod
    def _resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implementation-specific balancing logic."""
        raise NotImplementedError
