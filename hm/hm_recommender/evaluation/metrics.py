"""
Evaluation metrics for the recommender system.
"""

import numpy as np
import pandas as pd
from lightfm.evaluation import precision_at_k, auc_score
from typing import Dict, Any
from hm_recommender.utils.logger import setup_logger

logger = setup_logger("hm_recommender.evaluation")

def evaluate_model(model, test_interactions, item_features=None, k=10) -> Dict[str, float]:
    """
    Evaluate a trained model using standard metrics.
    
    Args:
        model: Trained LightFM model
        test_interactions: Test interactions matrix
        item_features: Item features matrix
        k: Number of items to consider for precision@k
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Calculate precision@k
    precision = precision_at_k(
        model, 
        test_interactions,
        item_features=item_features,
        k=k
    ).mean()
    
    # Calculate AUC
    auc = auc_score(
        model,
        test_interactions,
        item_features=item_features
    ).mean()
    
    metrics = {
        f'precision@{k}': float(precision),
        'auc': float(auc)
    }
    
    logger.info(f"Evaluation metrics: {metrics}")
    return metrics

def split_interactions_by_time(interactions_df: pd.DataFrame, test_fraction: float = 0.2) -> tuple:
    """
    Split interactions into training and test sets based on timestamp.
    
    Args:
        interactions_df: DataFrame containing user-item interactions
        test_fraction: Fraction of interactions to use for testing
        
    Returns:
        Tuple of (train_df, test_df)
    """
    # Sort interactions by timestamp
    interactions_df = interactions_df.sort_values('timestamp')
    
    # Split interactions
    split_idx = int(len(interactions_df) * (1 - test_fraction))
    train_df = interactions_df.iloc[:split_idx]
    test_df = interactions_df.iloc[split_idx:]
    
    logger.info(f"Split interactions into {len(train_df)} training and {len(test_df)} test interactions")
    return train_df, test_df

def calculate_diversity(recommendations: list, item_features: dict) -> float:
    """
    Calculate diversity of recommendations based on item features.
    
    Args:
        recommendations: List of recommendation dictionaries
        item_features: Dictionary mapping item IDs to feature dictionaries
        
    Returns:
        Diversity score (0-1, higher is more diverse)
    """
    if not recommendations or len(recommendations) < 2:
        return 0.0
        
    # Extract categories from recommendations
    categories = []
    for rec in recommendations:
        item_id = rec['item_id']
        if item_id in item_features:
            high_cat = item_features[item_id].get('high_category')
            specific_cat = item_features[item_id].get('specific_category')
            if high_cat and specific_cat:
                categories.append((high_cat, specific_cat))
    
    # Count unique categories
    unique_categories = set(categories)
    
    # Calculate diversity as ratio of unique categories to total recommendations
    diversity = len(unique_categories) / len(recommendations)
    
    return diversity 