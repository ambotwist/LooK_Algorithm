"""
Data module for the recommender system.
"""

from hm_recommender.data.fetcher import (
    fetch_items,
    fetch_interactions,
    fetch_item_details,
    fetch_disliked_items
)
from hm_recommender.data.processor import (
    create_user_item_mappings,
    process_item_features,
    build_interactions_matrix,
    extract_disliked_keywords
)

__all__ = [
    'fetch_items',
    'fetch_interactions',
    'fetch_item_details',
    'fetch_disliked_items',
    'create_user_item_mappings',
    'process_item_features',
    'build_interactions_matrix',
    'extract_disliked_keywords'
] 