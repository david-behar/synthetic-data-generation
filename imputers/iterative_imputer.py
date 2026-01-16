"""
Iterative Imputer implementation using MICE (Multivariate Imputation by Chained Equations).
"""

from sklearn.experimental import enable_iterative_imputer  # Explicitly enable
from sklearn.impute import IterativeImputer as SKLearnIterativeImputer
from .base import GenerativeImputer


class IterativeImputer(GenerativeImputer):
    """
    MICE (Multivariate Imputation by Chained Equations) implementation.
    
    Uses scikit-learn's IterativeImputer with BayesianRidge as the default estimator.
    Fast and works well for linear relationships in data.
    """
    
    def __init__(self, max_iter=10):
        """
        Initialize the Iterative Imputer.
        
        Args:
            max_iter: Maximum number of imputation rounds
        """
        super().__init__(max_iter=max_iter)
        
    def _impute(self, df_encoded):
        """
        Impute missing values using MICE algorithm.
        
        Args:
            df_encoded: Preprocessed DataFrame with encoded categorical variables
            
        Returns:
            Imputed matrix (numpy array)
        """
        matrix = df_encoded.values
        
        # MICE (BayesianRidge is default, fast and linear)
        imputer = SKLearnIterativeImputer(max_iter=self.max_iter, random_state=42)
        imputed_matrix = imputer.fit_transform(matrix)
        
        return imputed_matrix
