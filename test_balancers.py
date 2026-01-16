"""Example usage for balancer implementations."""

import numpy as np
import pandas as pd
from balancers import SmoteNCBalancer, AdasynBalancer, CTGANBalancer


def generate_mixed_data(n_samples: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            'amount': rng.normal(7500, 2000, n_samples).clip(100, 20000),
            'country': rng.choice(['US', 'MX', 'DE', 'SG'], n_samples),
            'currency': rng.choice(['USD', 'EUR', 'SGD', 'MXN'], n_samples),
            'channel': rng.choice(['web', 'branch', 'atm'], n_samples),
            'is_fraud': rng.choice([0, 1], size=n_samples, p=[0.93, 0.07]),
        }
    )
    return df


def generate_numeric_data(n_samples: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            'amount': rng.lognormal(mean=8, sigma=0.8, size=n_samples),
            'tenure_days': rng.integers(1, 365, size=n_samples),
            'velocity': rng.random(n_samples) * 10,
            'is_fraud': rng.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
        }
    )
    return df


def demo_smote_nc() -> None:
    df = generate_mixed_data()
    balancer = SmoteNCBalancer(target_col='is_fraud', strategy='auto')
    df_balanced = balancer.fit_resample(df)
    print(df_balanced.head())


def demo_adasyn() -> None:
    df = generate_numeric_data()
    balancer = AdasynBalancer(target_col='is_fraud', strategy='auto')
    df_balanced = balancer.fit_resample(df)
    print(df_balanced.head())


def demo_ctgan() -> None:
    df = generate_mixed_data()
    balancer = CTGANBalancer(target_col='is_fraud', epochs=50, verbose=False)
    df_balanced = balancer.fit_resample(df)
    print(df_balanced.tail())


def run_all_demos() -> None:
    print('\n=== SMOTE-NC Demo ===')
    demo_smote_nc()
    print('\n=== ADASYN Demo ===')
    demo_adasyn()
    print('\n=== CTGAN Demo ===')
    demo_ctgan()


if __name__ == '__main__':
    run_all_demos()
