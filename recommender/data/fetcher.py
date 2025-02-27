"""
Data fetching utilities for the recommender system.
"""

import pandas as pd
from supabase import Client
import logging
from recommender.utils.logger import setup_logger

logger = setup_logger("hm_recommender.data")

def fetch_items(supabase: Client) -> pd.DataFrame:
    """
    Fetch all clothing items from the hm_items table using Supabase.

    Args:
        supabase: Initialized Supabase client
        
    Returns:
        DataFrame containing item data
        
    Raises:
        Exception: If there's an error fetching the data
    """
    try:
        # Fetch items from Supabase
        response = supabase.table('hm_items').select(
            'id, brand, price, high_category, specific_category, ' +
            'colors, styles, materials, sex, condition, season, ' +
            'top_size, bottom_size, shoe_size, fit, name, images'
        ).eq('is_active', True).execute()
        
        # Convert to DataFrame
        items_df = pd.DataFrame(response.data)
        logger.info(f"Fetched {len(items_df)} items from database")
        return items_df
    except Exception as e:
        logger.error(f"Error fetching items: {e}")
        raise
    
def fetch_interactions(supabase: Client) -> pd.DataFrame:
    """
    Fetch user-item interactions (likes, dislikes, superlikes) using Supabase.
    
    Args:
        supabase: Initialized Supabase client
        
    Returns:
        DataFrame containing user interactions
        
    Raises:
        ValueError: If the interactions table doesn't exist or has an unexpected structure
    """
    try:
        # Fetch interactions from Supabase
        response = supabase.table('hm_interactions').select(
            'user_id, item_id, interaction_type, created_at'
        ).execute()
        
        # Check if interactions table is empty
        if not response.data:
            error_msg = "No interactions found in the database. The recommender system requires user interactions to provide personalized recommendations."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Convert to DataFrame
        interactions_df = pd.DataFrame(response.data)
        
        # Map interaction types to weights
        interactions_df['weight'] = interactions_df['interaction_type'].map({
            'like': 1,
            'superlike': 2,
            'dislike': -1
        }).fillna(0)
        
        # Use created_at as timestamp
        interactions_df['timestamp'] = interactions_df['created_at']
        
        # Sort by timestamp
        interactions_df = interactions_df.sort_values('timestamp', ascending=False)
            
        logger.info(f"Fetched {len(interactions_df)} interactions from database")
        return interactions_df
    except Exception as e:
        logger.error(f"Error fetching interactions: {e}")
        # If the table doesn't exist yet, raise a clear error
        if "does not exist" in str(e):
            error_msg = "Interactions table does not exist. Please create the hm_interactions table before running the recommender."
            logger.error(error_msg)
            raise ValueError(error_msg)
        raise
        
def fetch_disliked_items(supabase: Client, user_id: str) -> list:
    """
    Fetch items that a user has disliked.
    
    Args:
        supabase: Initialized Supabase client
        user_id: ID of the user
        
    Returns:
        List of disliked item IDs
    """
    try:
        response = supabase.table('hm_interactions').select('item_id').eq('user_id', user_id).eq('interaction_type', 'dislike').execute()
        return [item['item_id'] for item in response.data]
    except Exception as e:
        logger.error(f"Error fetching disliked items: {e}")
        return []
        
def fetch_item_details(supabase: Client, item_ids: list) -> dict:
    """
    Fetch details for specific items.
    
    Args:
        supabase: Initialized Supabase client
        item_ids: List of item IDs to fetch details for
        
    Returns:
        Dictionary mapping item IDs to their details
    """
    if not item_ids:
        return {}
        
    try:
        response = supabase.table('hm_items').select(
            'id, brand, price, high_category, specific_category, colors, styles, name, images'
        ).in_('id', item_ids).execute()
        
        # Log the number of items fetched
        logger.info(f"Fetched details for {len(response.data)} items")
        
        # Return a dictionary mapping item IDs to their details
        return {item['id']: item for item in response.data}
    except Exception as e:
        logger.error(f"Error fetching item details: {e}")
        return {} 