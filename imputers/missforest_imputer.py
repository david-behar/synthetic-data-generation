"""
MissForest Imputer implementation using Random Forest.
"""

from sklearn.experimental import enable_iterative_imputer  # Explicitly enable
from sklearn.impute import IterativeImputer as SKLearnIterativeImputer
from sklearn.ensemble import RandomForestRegressor
from .base import GenerativeImputer


class MissForestImputer(GenerativeImputer):
    """
    MissForest implementation using Random Forest regressors.
    
    Uses scikit-learn's IterativeImputer with RandomForestRegressor as the estimator.
    Handles non-linear relationships and interaction effects well.
    Best for mixed numerical and categorical data.
    """
    
    def __init__(self, max_iter=10, n_estimators=50, max_depth=10):
        """
        Initialize the MissForest Imputer.
        
        Args:
            max_iter: Maximum number of imputation rounds
            n_estimators: Number of trees in the random forest
            max_depth: Maximum depth of trees
        """
        super().__init__(max_iter=max_iter)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
    def _impute(self, df_encoded):
        """
        Impute missing values using MissForest algorithm (Random Forest based).
        
        Args:
            df_encoded: Preprocessed DataFrame with encoded categorical variables
            
        Returns:
            Imputed matrix (numpy array)
        """
        matrix = df_encoded.values
        
        # MissForest uses Random Forest which handles non-linearities well
        estimator = RandomForestRegressor(
            n_jobs=-1, 
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=42
        )
        
        imputer = SKLearnIterativeImputer(
            estimator=estimator, 
            max_iter=self.max_iter, 
            random_state=42
        )
        imputed_matrix = imputer.fit_transform(matrix)
        
        return imputed_matrix
