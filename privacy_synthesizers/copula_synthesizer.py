"""Gaussian Copula based privacy synthesizer."""

from __future__ import annotations

from typing import Any, Dict

from sdv.single_table import GaussianCopulaSynthesizer

from .base import PrivacySynthesizer


class CopulaPrivacySynthesizer(PrivacySynthesizer):
    """Fast statistical synthesizer using Gaussian Copulas."""

    def __init__(self, pii_columns=None, model_kwargs: Dict[str, Any] | None = None) -> None:
        super().__init__(pii_columns=pii_columns)
        self.model_kwargs: Dict[str, Any] = model_kwargs or {}

    def _build_model(self, metadata):
        return GaussianCopulaSynthesizer(metadata, **self.model_kwargs)
