"""Compatibility helpers for the privacy synthesizer hierarchy."""

from __future__ import annotations

from typing import Any, Dict, Type

from privacy_synthesizers import (
    CTGANPrivacySynthesizer,
    CopulaPrivacySynthesizer,
    PrivacySynthesizer,
)


SYNTH_REGISTRY: Dict[str, Type[PrivacySynthesizer]] = {
    'copula': CopulaPrivacySynthesizer,
    'ctgan': CTGANPrivacySynthesizer,
}


def create_synthesizer(method: str = 'copula', **kwargs: Any) -> PrivacySynthesizer:
    """Factory helper mirroring the legacy interface."""
    try:
        cls = SYNTH_REGISTRY[method]
    except KeyError as exc:
        raise ValueError("Method must be 'copula' or 'ctgan'.") from exc
    return cls(**kwargs)


def fit_sample(df, method: str = 'copula', n_samples: int | None = None, **kwargs: Any):
    """Convenience wrapper to match the previous module-level function."""
    synth = create_synthesizer(method=method, **kwargs)
    return synth.fit_sample(df, n_samples=n_samples)


if __name__ == '__main__':
    from test_privacy_synthesizers import run_all_demos

    run_all_demos()