"""
Autoencoder Imputer implementation using deep learning.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from .base import GenerativeImputer


class AutoencoderImputer(GenerativeImputer):
    """
    Denoising Autoencoder implementation for imputation.
    
    Uses a deep learning autoencoder to learn the data manifold and 
    reconstruct missing values. Particularly effective for complex,
    high-dimensional data with non-linear patterns.
    """
    
    def __init__(self, max_iter=10, epochs=50, batch_size=32):
        """
        Initialize the Autoencoder Imputer.
        
        Args:
            max_iter: Not used in autoencoder (kept for API consistency)
            epochs: Number of training epochs for the neural network
            batch_size: Batch size for training
        """
        super().__init__(max_iter=max_iter)
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = None
        
    def _impute(self, df_encoded):
        """
        Impute missing values using a Denoising Autoencoder.
        
        Args:
            df_encoded: Preprocessed DataFrame with encoded categorical variables
            
        Returns:
            Imputed matrix (numpy array)
        """
        data = df_encoded.values
        
        # Scale data to [0, 1] for Neural Net stability
        self.scaler = MinMaxScaler()
        # Impute temporarily with Mean just to feed into scaler
        temp_fill = pd.DataFrame(data).fillna(df_encoded.mean()).values
        data_scaled = self.scaler.fit_transform(temp_fill)
        
        # Create Mask (1 = Present, 0 = Missing)
        mask = ~np.isnan(data)
        
        # Build Autoencoder Architecture
        input_dim = data.shape[1]
        inputs = layers.Input(shape=(input_dim,))
        
        # Encoder
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dropout(0.2)(x)  # Noise injection
        x = layers.Dense(32, activation='relu')(x)
        
        # Bottleneck (Latent Space)
        encoded = layers.Dense(16, activation='relu')(x)
        
        # Decoder
        x = layers.Dense(32, activation='relu')(encoded)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(input_dim, activation='sigmoid')(x)
        
        autoencoder = models.Model(inputs, outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Training
        print("Training Autoencoder...")
        autoencoder.fit(
            data_scaled, data_scaled,  # Self-supervision
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            verbose=0
        )
        
        # Predict
        reconstructed_scaled = autoencoder.predict(data_scaled, verbose=0)
        reconstructed = self.scaler.inverse_transform(reconstructed_scaled)
        
        # Fill only missing values
        final_data = data.copy()
        # Where data was NaN, use reconstructed value
        final_data[~mask] = reconstructed[~mask]
        
        return final_data
