"""
Base class for Generative Imputation methods.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from pandas.api.types import is_datetime64_any_dtype, is_datetime64tz_dtype
from sklearn.preprocessing import OrdinalEncoder


class GenerativeImputer(ABC):
    """
    Abstract base class for State-of-the-Art Generative Imputation.
    
    All concrete imputers must implement the _impute() method.
    The base class handles:
    - Encoding/decoding of categorical variables
    - Pre/post-processing pipeline
    - Common utilities
    """
    
    def __init__(self, max_iter=10):
        """
        Initialize the imputer.
        
        Args:
            max_iter: Maximum number of iterations for iterative methods
        """
        self.max_iter = max_iter
        self.encoders = {}
        self.scalers = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.datetime_cols = []
        
    def _preprocess(self, df):
        """
        Handles encoding of categorical variables so algorithms can process them.
        
        Args:
            df: Input DataFrame with missing values
            
        Returns:
            Encoded DataFrame with numerical representations of categorical data
        """
        self.encoders = {}
        self.numerical_cols = []
        self.categorical_cols = []
        self.datetime_cols = []

        # Track datetime columns so the imputation algorithms can safely skip them
        for col in df.columns:
            series = df[col]
            if is_datetime64_any_dtype(series) or is_datetime64tz_dtype(series):
                self.datetime_cols.append(col)

        # Work on a copy that excludes datetime/timestamp columns entirely
        df_encoded = df.drop(columns=self.datetime_cols, errors='ignore').copy()
        
        # Identify types on the reduced frame
        self.numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
        
        # Encode Categoricals (Ordinal Encoding)
        # Strategy: We temporarily handle NaNs, encode, then put NaNs back.
        for col in self.categorical_cols:
            # Placeholder for NaN
            series = df[col].astype(str)
            mask = df[col].isna()
            
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            # Fit on non-NaN data
            non_nan_data = series[~mask].values.reshape(-1, 1)
            encoder.fit(non_nan_data)
            
            # Transform everything
            encoded_col = encoder.transform(series.values.reshape(-1, 1))
            
            # Re-inject NaNs
            encoded_col[mask] = np.nan
            df_encoded[col] = encoded_col
            self.encoders[col] = encoder
            
        return df_encoded

    def _inverse_transform(self, df_imputed):
        """
        Converts numerical encodings back to original strings.
        
        Args:
            df_imputed: DataFrame with imputed numerical values
            
        Returns:
            DataFrame with categorical columns decoded to original format
        """
        df_out = df_imputed.copy()
        # Round categorical columns to nearest integer (as model might output 1.2 instead of 1)
        for col in self.categorical_cols:
            if col in self.encoders:
                encoder = self.encoders[col]
                # Clip to valid range
                df_out[col] = df_out[col].round().clip(0, len(encoder.categories_[0])-1)
                df_out[col] = encoder.inverse_transform(df_out[[col]])

        return df_out
        return df_out

    @abstractmethod
    def _impute(self, df_encoded):
        """
        Core imputation logic to be implemented by subclasses.
        
        Args:
            df_encoded: Preprocessed DataFrame with encoded categorical variables
            
        Returns:
            Imputed matrix (numpy array)
        """
        pass

    def fit_transform(self, df):
        """
        Main entry point: preprocess, impute, and postprocess data.
        
        Args:
            df: Input DataFrame with missing values
            
        Returns:
            DataFrame with all missing values imputed
        """
        print(f"--- Starting Imputation using {self.__class__.__name__} ---")
        
        # 1. Preprocess (Encode Categoricals)
        df_encoded = self._preprocess(df)
        
        # 2. Apply Algorithm (implemented by subclass)
        imputed_matrix = self._impute(df_encoded)
        
        # 3. Reconstruct DataFrame
        df_imputed = pd.DataFrame(imputed_matrix, columns=df_encoded.columns, index=df.index)
        
        # 4. Inverse Transform (Decode Categoricals)
        df_final = self._inverse_transform(df_imputed)

        # 5. Add back datetime columns (left untouched) and ensure original column order
        for col in self.datetime_cols:
            df_final[col] = df[col]
        df_final = df_final[df.columns]
        
        return df_final
