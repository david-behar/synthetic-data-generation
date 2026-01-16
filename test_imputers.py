"""
Test and example usage of imputers.
Demonstrates how to use different imputation strategies.
"""

import numpy as np
import pandas as pd
from imputers import MissForestImputer, IterativeImputer, AutoencoderImputer


def generate_sample_data():
    """
    Generate sample data with missing values for testing.
    You should replace this with your actual data generation function.
    """
    np.random.seed(42)
    
    # Create sample dataframe
    n_samples = 100
    data = {
        'energy_kwh': np.random.uniform(50, 200, n_samples),
        'temperature': np.random.uniform(20, 80, n_samples),
        'pressure': np.random.uniform(1, 5, n_samples),
        'production_mode': np.random.choice(['Normal', 'Turbo', 'Eco'], n_samples)
    }
    df = pd.DataFrame(data)
    
    # Introduce missing values
    missing_mask = np.random.random((n_samples, 4)) < 0.2
    for i, col in enumerate(df.columns):
        df.loc[missing_mask[:, i], col] = np.nan
    
    return df


def test_missforest():
    """Test MissForest imputer."""
    print("\n" + "="*60)
    print("Testing MissForest Imputer")
    print("="*60)
    
    df = generate_sample_data()
    print("\n--- Original Missing Count ---")
    print(df.isnull().sum())
    
    imputer = MissForestImputer(max_iter=5, n_estimators=50, max_depth=10)
    df_clean = imputer.fit_transform(df)
    
    print("\n--- Imputed Missing Count (Should be 0) ---")
    print(df_clean.isnull().sum())
    
    print("\n--- Sample of Imputed Data ---")
    print(df_clean.head(10))


def test_iterative():
    """Test Iterative (MICE) imputer."""
    print("\n" + "="*60)
    print("Testing Iterative (MICE) Imputer")
    print("="*60)
    
    df = generate_sample_data()
    print("\n--- Original Missing Count ---")
    print(df.isnull().sum())
    
    imputer = IterativeImputer(max_iter=10)
    df_clean = imputer.fit_transform(df)
    
    print("\n--- Imputed Missing Count (Should be 0) ---")
    print(df_clean.isnull().sum())
    
    print("\n--- Sample of Imputed Data ---")
    print(df_clean.head(10))


def test_autoencoder():
    """Test Autoencoder imputer."""
    print("\n" + "="*60)
    print("Testing Autoencoder Imputer")
    print("="*60)
    
    df = generate_sample_data()
    print("\n--- Original Missing Count ---")
    print(df.isnull().sum())
    
    imputer = AutoencoderImputer(epochs=50, batch_size=32)
    df_clean = imputer.fit_transform(df)
    
    print("\n--- Imputed Missing Count (Should be 0) ---")
    print(df_clean.isnull().sum())
    
    print("\n--- Sample of Imputed Data ---")
    print(df_clean.head(10))


def compare_all_methods():
    """Compare all three imputation methods on the same dataset."""
    print("\n" + "="*60)
    print("Comparing All Imputation Methods")
    print("="*60)
    
    # Generate same data for fair comparison
    df_original = generate_sample_data()
    
    methods = {
        'MissForest': MissForestImputer(max_iter=5),
        'Iterative': IterativeImputer(max_iter=10),
        'Autoencoder': AutoencoderImputer(epochs=30, batch_size=32)
    }
    
    results = {}
    for name, imputer in methods.items():
        print(f"\nProcessing with {name}...")
        df_clean = imputer.fit_transform(df_original.copy())
        results[name] = df_clean
    
    print("\n--- Comparison Summary ---")
    for name, df_clean in results.items():
        missing_count = df_clean.isnull().sum().sum()
        print(f"{name}: {missing_count} missing values remaining")


if __name__ == "__main__":
    # Run individual tests
    test_missforest()
    test_iterative()
    test_autoencoder()
    
    # Compare all methods
    compare_all_methods()
