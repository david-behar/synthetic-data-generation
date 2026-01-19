import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class UniversalDataGenerator:
    """
    A generic engine to generate synthetic data based on a user-defined schema.
    Supports:
    - Statistical Distributions (Normal, Uniform, Integer)
    - Categorical Sampling (Weighted)
    - Date Ranges
    - Formula-based Columns (for creating relationships and targets)
    """
    
    def __init__(self, n_rows=1000, seed=42):
        self.n_rows = n_rows
        self.seed = seed
        self.df = pd.DataFrame()
        np.random.seed(seed)
        random.seed(seed)

    def generate_independent(self, name, type, **kwargs):
        """
        Generates a standalone column based on a distribution.
        
        :param type: 'normal', 'uniform', 'integer', 'category', 'date'
        """
        if type == 'normal':
            mean = kwargs.get('mean', 0)
            std = kwargs.get('std', 1)
            self.df[name] = np.random.normal(mean, std, self.n_rows)
            
        elif type == 'uniform':
            low = kwargs.get('min', 0)
            high = kwargs.get('max', 1)
            self.df[name] = np.random.uniform(low, high, self.n_rows)
            
        elif type == 'integer':
            low = kwargs.get('min', 0)
            high = kwargs.get('max', 100)
            self.df[name] = np.random.randint(low, high, self.n_rows)
            
        elif type == 'category':
            choices = kwargs.get('choices', ['A', 'B'])
            weights = kwargs.get('weights', None) # List of probs summing to 1
            self.df[name] = np.random.choice(choices, self.n_rows, p=weights)
            
        elif type == 'date':
            start_str = kwargs.get('start', '2023-01-01')
            end_str = kwargs.get('end', '2024-01-01')
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d")
            delta = end - start
            days_range = delta.days
            
            # Add random days to start date
            random_days = np.random.randint(0, days_range, self.n_rows)
            self.df[name] = [start + timedelta(days=int(d)) for d in random_days]
            
        return self

    def generate_dependent(self, name, formula, noise_std=0.0):
        """
        Generates a column based on a formula using existing columns.
        This is how you create TARGET variables or correlations.
        
        :param formula: String expression using pandas syntax (e.g. "col_A * 2 + col_B")
        :param noise_std: Add random Gaussian noise to make prediction harder.
        """
        # We use pandas eval for efficient vector calculation
        try:
            # Evaluate the formula in the context of the dataframe
            result = self.df.eval(formula)
            
            # Add noise if requested (only for numerical)
            if noise_std > 0 and pd.api.types.is_numeric_dtype(result):
                noise = np.random.normal(0, noise_std, self.n_rows)
                result = result + noise
                
            self.df[name] = result
            
        except Exception as e:
            print(f"Error generating column '{name}': {e}")
            
        return self
    
    def generate_conditional(self, name, condition_col, mapping_dict, default=None):
        """
        Generates values based on another column's value (e.g., If Country=US, Currency=USD).
        """
        # Create a map function
        def mapper(val):
            return mapping_dict.get(val, default if default else val)
            
        self.df[name] = self.df[condition_col].map(mapper)
        return self

    def get_dataframe(self):
        return self.df

# ==========================================
# EXAMPLE USE CASE: Real Estate Price Prediction
# ==========================================
# We want to create a dataset where 'House_Price' is the target.
# It should depend on Square_Feet, Location, and Age.

if __name__ == "__main__":
    
    # 1. Initialize Generator
    gen = UniversalDataGenerator(n_rows=5000)
    
    # 2. Define Independent Variables (The Features)
    gen.generate_independent('Square_Feet', 'integer', min=500, max=3500)
    gen.generate_independent('House_Age', 'integer', min=0, max=100)
    gen.generate_independent('Location_Score', 'normal', mean=5, std=1.5) # 1-10 scale essentially
    gen.generate_independent('Has_Pool', 'category', choices=[0, 1], weights=[0.8, 0.2])
    
    # 3. Define Dependent Variables (Correlations)
    
    # Distance to City Center (Negative correlation with Location Score for this example)
    # "If location score is high, distance is low"
    gen.generate_dependent('Distance_Miles', '15 - Location_Score + 2', noise_std=1.0)
    
    # 4. Define The TARGET Variable (The Formula)
    # Logic: Price = (SqFt * 200) - (Age * 1000) + (Location * 10000) + (Pool * 20000)
    
    price_formula = """
    (Square_Feet * 200) - (House_Age * 1000) + (Location_Score * 10000) + (Has_Pool * 25000)
    """
    
    # We add considerable noise (std=15000) so the model doesn't get 100% accuracy
    gen.generate_dependent('Price', price_formula, noise_std=15000)
    
    # 5. Output
    df = gen.get_dataframe()
    
    # --- Verification ---
    print("--- Generated Real Estate Data ---")
    print(df.head())
    
    print("\n--- Checking Predictability ---")
    print(f"Correlation (SqFt vs Price): {df['Square_Feet'].corr(df['Price']):.2f} (Should be high positive)")
    print(f"Correlation (Age vs Price): {df['House_Age'].corr(df['Price']):.2f} (Should be negative)")