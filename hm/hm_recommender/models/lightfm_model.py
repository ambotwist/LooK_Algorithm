"""
LightFM model implementation for the recommender system.
"""

import numpy as np
from lightfm import LightFM
import scipy.sparse as sparse
from typing import Dict, Any, Optional
from hm_recommender.utils.logger import setup_logger

logger = setup_logger("hm_recommender.model")

class LightFMRecommender:
    """
    LightFM-based recommender model.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the LightFM recommender.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        
    def train(self, interactions: sparse.coo_matrix, item_features: sparse.csr_matrix = None,
             num_components: int = 30, learning_rate: float = 0.05, 
             epochs: int = 20, loss: str = 'warp') -> None:
        """
        Train the LightFM recommendation model.
        
        Args:
            interactions: User-item interaction matrix
            item_features: Item features matrix
            num_components: Number of latent factors (embeddings dimension)
            learning_rate: Learning rate for the training algorithm
            epochs: Number of training epochs
            loss: Loss function ('warp', 'bpr', 'logistic', or 'warp-kos')
                  - 'warp': Weighted Approximate-Rank Pairwise (recommended for implicit feedback)
                  - 'bpr': Bayesian Personalized Ranking (for implicit feedback)
                  - 'logistic': Logistic loss (for explicit feedback)
                  - 'warp-kos': k-Order Statistic loss (for diversity)
        """
        # Initialize the model
        self.model = LightFM(
            no_components=num_components,
            learning_rate=learning_rate,
            loss=loss,
            random_state=self.random_state
        )
        
        # Train the model
        self.model.fit(
            interactions=interactions,
            item_features=item_features,
            epochs=epochs,
            verbose=True
        )
        
        logger.info(f"Trained model with {num_components} components using {loss} loss")
        
    def predict(self, user_ids: np.ndarray, item_ids: np.ndarray, 
               item_features: sparse.csr_matrix = None) -> np.ndarray:
        """
        Generate predictions for user-item pairs.
        
        Args:
            user_ids: User IDs to generate predictions for
            item_ids: Item IDs to generate predictions for
            item_features: Item features matrix
            
        Returns:
            Array of prediction scores
            
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return self.model.predict(
            user_ids=user_ids,
            item_ids=item_ids,
            item_features=item_features
        )
        
    def get_item_representations(self) -> Dict[str, np.ndarray]:
        """
        Get the learned item representations (embeddings).
        
        Returns:
            Dictionary containing item biases and embeddings
            
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return {
            'item_biases': self.model.item_biases,
            'item_embeddings': self.model.item_embeddings
        }
        
    def get_user_representations(self) -> Dict[str, np.ndarray]:
        """
        Get the learned user representations (embeddings).
        
        Returns:
            Dictionary containing user biases and embeddings
            
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        return {
            'user_biases': self.model.user_biases,
            'user_embeddings': self.model.user_embeddings
        }
        
    def save_model(self, filepath: str) -> None:
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model to
            
        Raises:
            ValueError: If the model has not been trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        # TODO: Implement model saving
        logger.info(f"Model saving not implemented yet")
        
    def load_model(self, filepath: str) -> None:
        """
        Load the model from a file.
        
        Args:
            filepath: Path to load the model from
        """
        # TODO: Implement model loading
        logger.info(f"Model loading not implemented yet") 