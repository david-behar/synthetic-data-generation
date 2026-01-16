"""CTGAN based privacy synthesizer."""

from __future__ import annotations

from typing import Any, Dict

from sdv.single_table import CTGANSynthesizer

from .base import PrivacySynthesizer


class CTGANPrivacySynthesizer(PrivacySynthesizer):
    """Deep learning synthesizer leveraging CTGAN for complex tables."""

    def __init__(
        self,
        pii_columns=None,
        epochs: int = 100,
        verbose: bool = True,
        model_kwargs: Dict[str, Any] | None = None,
    ) -> None:
        super().__init__(pii_columns=pii_columns)
        self.epochs = epochs
        self.verbose = verbose
        self.model_kwargs: Dict[str, Any] = model_kwargs or {}

    def _build_model(self, metadata):
        return CTGANSynthesizer(
            metadata,
            epochs=self.epochs,
            verbose=self.verbose,
            **self.model_kwargs,
        )
