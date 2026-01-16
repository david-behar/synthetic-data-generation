"""Compatibility wrapper for the new balancers package."""

from __future__ import annotations

from typing import Any, Dict, Type

from balancers import AdasynBalancer, CTGANBalancer, SmoteNCBalancer, SyntheticBalancer


BALANCER_REGISTRY: Dict[str, Type[SyntheticBalancer]] = {
    'smote_nc': SmoteNCBalancer,
    'adasyn': AdasynBalancer,
    'ctgan': CTGANBalancer,
}


def create_balancer(target_col: str, method: str = 'smote_nc', **kwargs: Any) -> SyntheticBalancer:
    """Factory helper to mirror the legacy interface."""
    try:
        balancer_cls = BALANCER_REGISTRY[method]
    except KeyError as exc:
        raise ValueError("Method must be 'smote_nc', 'adasyn', or 'ctgan'.") from exc
    return balancer_cls(target_col=target_col, **kwargs)


def fit_resample(df, target_col: str, method: str = 'smote_nc', **kwargs: Any):
    """Convenience wrapper matching the legacy free function API."""
    balancer = create_balancer(target_col=target_col, method=method, **kwargs)
    return balancer.fit_resample(df)


if __name__ == '__main__':
    from test_balancers import run_all_demos

    run_all_demos()