"""
Evaluation module for the recommender system.
"""

from hm_recommender.evaluation.metrics import (
    evaluate_model,
    split_interactions_by_time,
    calculate_diversity
)

__all__ = [
    'evaluate_model',
    'split_interactions_by_time',
    'calculate_diversity'
] 