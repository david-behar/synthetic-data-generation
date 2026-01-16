"""Compatibility wrapper for the refactored imputers package."""

from __future__ import annotations

from typing import Any, Dict, Type

from imputers import (
    AutoencoderImputer,
    GenerativeImputer,
    IterativeImputer,
    MissForestImputer,
)


IMPUTER_REGISTRY: Dict[str, Type[GenerativeImputer]] = {
    'iterative': IterativeImputer,
    'missforest': MissForestImputer,
    'autoencoder': AutoencoderImputer,
}


def create_imputer(method: str = 'missforest', **kwargs: Any) -> GenerativeImputer:
    """Factory helper preserving the legacy interface."""
    try:
        imputer_cls = IMPUTER_REGISTRY[method]
    except KeyError as exc:
        raise ValueError("Method must be 'iterative', 'missforest', or 'autoencoder'.") from exc
    return imputer_cls(**kwargs)


def fit_transform(df, method: str = 'missforest', **kwargs: Any):
    """Convenience wrapper mirroring the original module-level call."""
    imputer = create_imputer(method=method, **kwargs)
    return imputer.fit_transform(df)


if __name__ == '__main__':
    from test_imputers import test_missforest, test_iterative, test_autoencoder, compare_all_methods

    test_missforest()
    test_iterative()
    test_autoencoder()
    compare_all_methods()
