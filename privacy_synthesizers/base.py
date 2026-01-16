"""Base class for privacy-preserving synthetic data generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

import pandas as pd
from sdv.metadata import SingleTableMetadata


class PrivacySynthesizer(ABC):
    """Common workflow for privacy-focused synthetic generators."""

    def __init__(self, pii_columns: Sequence[str] | None = None) -> None:
        self.pii_columns: List[str] = list(pii_columns) if pii_columns else []
        self.metadata: SingleTableMetadata | None = None
        self.model = None

    def fit_sample(self, df: pd.DataFrame, n_samples: int | None = None) -> pd.DataFrame:
        """Sanitize input, train the model, and sample synthetic rows."""
        print(f"\n--- Starting Privacy Synthesis using {self.__class__.__name__} ---")
        df_safe = self._detect_and_drop_pii(df)
        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(df_safe)
        self.model = self._build_model(self.metadata)
        self.model.fit(df_safe)
        count = n_samples if n_samples is not None else len(df)
        print(f"Generating {count} synthetic rows...")
        return self.model.sample(num_rows=count)

    def _detect_and_drop_pii(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply explicit and heuristic PII stripping prior to training."""
        df_clean = df.copy()
        if self.pii_columns:
            print(f"Dropping user-specified PII columns: {self.pii_columns}")
            df_clean = df_clean.drop(columns=self.pii_columns, errors='ignore')
        keywords = ['name', 'ssn', 'social', 'phone', 'email', 'address', 'id']
        cols_to_drop: List[str] = []
        for col in df_clean.columns:
            lowered = col.lower()
            if any(keyword in lowered for keyword in keywords) and col not in ['patient_id', 'employee_id']:
                cols_to_drop.append(col)
        if cols_to_drop:
            print(f"Auto-detected and dropping potential PII: {cols_to_drop}")
            df_clean = df_clean.drop(columns=cols_to_drop)
        return df_clean

    @abstractmethod
    def _build_model(self, metadata: SingleTableMetadata):
        """Return an initialized SDV synthesizer."""
        raise NotImplementedError
