"""SMOTE-NC based balancer."""

from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTENC
from .base import SyntheticBalancer


class SmoteNCBalancer(SyntheticBalancer):
    """SMOTE-NC implementation for datasets with categorical columns."""

    def _resample(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        cat_indices = self._get_categorical_indices(X)
        smote = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=self.strategy,
            random_state=42,
        )
        X_res, y_res = smote.fit_resample(X, y)
        balanced = pd.concat(
            [pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=self.target_col)],
            axis=1,
        )
        original_len = len(df)
        mask = pd.Series(False, index=balanced.index)
        mask.iloc[original_len:] = True
        return balanced, mask

    def _get_categorical_indices(self, X: pd.DataFrame) -> list[int]:
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        return [X.columns.get_loc(col) for col in cat_cols]
