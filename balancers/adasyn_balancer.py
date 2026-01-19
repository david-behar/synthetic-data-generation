"""ADASYN based balancer."""

from __future__ import annotations

import warnings
import pandas as pd
from imblearn.over_sampling import ADASYN
from .base import SyntheticBalancer


class AdasynBalancer(SyntheticBalancer):
    """ADASYN implementation for numerical datasets."""

    def _resample(self, df: pd.DataFrame):
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        X_num = X.select_dtypes(include=['number'])
        if X_num.shape[1] != X.shape[1]:
            warnings.warn(
                "ADASYN received categorical columns and will ignore them. Use SmoteNC for mixed data.",
                UserWarning,
            )
        adasyn = ADASYN(sampling_strategy=self.strategy, random_state=42)
        X_res, y_res = adasyn.fit_resample(X_num, y)
        balanced = pd.concat(
            [pd.DataFrame(X_res, columns=X_num.columns), pd.Series(y_res, name=self.target_col)],
            axis=1,
        )
        original_len = len(df)
        mask = pd.Series(False, index=balanced.index)
        mask.iloc[original_len:] = True
        return balanced, mask
