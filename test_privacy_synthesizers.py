"""Demo utilities for privacy synthesizers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from privacy_synthesizers import CTGANPrivacySynthesizer, CopulaPrivacySynthesizer


def generate_sensitive_healthcare(n_samples: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            'Patient_Name': [f'Patient_{i}' for i in range(n_samples)],
            'SSN': [f'{rng.integers(100, 999)}-\d\d-\d\d\d\d' for _ in range(n_samples)],
            'Phone_Number': [f'+1-555-{rng.integers(1000, 9999)}' for _ in range(n_samples)],
            'Zip_Code': rng.choice(['94107', '10001', '75201', '60601'], n_samples),
            'Gender': rng.choice(['M', 'F'], n_samples),
            'Primary_Diagnosis': rng.choice(['Oncology', 'Cardiology', 'Orthopedics'], n_samples),
            'Total_Bill_Amount': rng.normal(25000, 5000, n_samples).clip(2000, 80000),
            'HIV_Status': rng.choice(['Positive', 'Negative'], p=[0.1, 0.9], size=n_samples),
        }
    )
    return df


def demo_copula() -> None:
    df = generate_sensitive_healthcare()
    synth = CopulaPrivacySynthesizer(pii_columns=['Patient_Name', 'SSN', 'Phone_Number'])
    synthetic = synth.fit_sample(df)
    print('\n--- Copula synthetic preview ---')
    print(synthetic.head())


def demo_ctgan() -> None:
    df = generate_sensitive_healthcare()
    synth = CTGANPrivacySynthesizer(
        pii_columns=['Patient_Name', 'SSN', 'Phone_Number'],
        epochs=50,
        verbose=False,
    )
    synthetic = synth.fit_sample(df, n_samples=300)
    print('\n--- CTGAN synthetic preview ---')
    print(synthetic.head())


def run_all_demos() -> None:
    print('\n=== Copula Demo ===')
    demo_copula()
    print('\n=== CTGAN Demo ===')
    demo_ctgan()


if __name__ == '__main__':
    run_all_demos()
