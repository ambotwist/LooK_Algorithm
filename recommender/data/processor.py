"""
Data processing utilities for the recommender system.
"""

import pandas as pd
import numpy as np
import scipy.sparse as sparse
from lightfm.data import Dataset
from typing import Dict, Tuple, List, Any
from recommender.utils.logger import setup_logger

logger = setup_logger("hm_recommender.processor")

def create_user_item_mappings(interactions_df: pd.DataFrame, items_df: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Create mappings between external IDs and internal indices for users and items.
    
    Args:
        interactions_df: DataFrame containing user-item interactions
        items_df: DataFrame containing item data
        
    Returns:
        Tuple of (user_id_map, item_id_map)
    """
    user_id_map = {}
    item_id_map = {}
    
    # Create user mapping
    for i, user_id in enumerate(interactions_df['user_id'].unique()):
        user_id_map[user_id] = i
        
    # Create item mapping for all items
    for i, item_id in enumerate(items_df['id']):
        item_id_map[item_id] = i
        
    logger.info(f"Created user mapping with {len(user_id_map)} users")
    logger.info(f"Created item mapping with {len(item_id_map)} items")
    
    return user_id_map, item_id_map

def process_item_features(items_df: pd.DataFrame, item_id_map: Dict[str, int]) -> Tuple[sparse.csr_matrix, Dataset]:
    """
    Process item features into a format suitable for LightFM.
    
    Args:
        items_df: DataFrame containing item data
        item_id_map: Mapping from external item IDs to internal indices
        
    Returns:
        Tuple of (item_features_matrix, dataset)
    """
    # Extract and process categorical features
    feature_list = []

    # Process high_category
    for category in items_df['high_category'].unique():
        if pd.notna(category):
            feature_list.append(f"high_category:{category}")

    # Process specific_category
    for category in items_df['specific_category'].unique():
        if pd.notna(category):
            feature_list.append(f"specific_category:{category}")
        
    # Process brand
    for brand in items_df['brand'].unique():
        if pd.notna(brand):
            feature_list.append(f"brand:{brand}")

    # Process colors (which is an array in PostgreSQL)
    all_colors = set()
    for colors in items_df['colors']:
        if isinstance(colors, list):
            all_colors.update(colors)
    for color in all_colors:
        feature_list.append(f"color:{color}")

    # Process styles (which is an array in PostgreSQL)
    all_styles = set()
    for styles in items_df['styles']:
        if isinstance(styles, list):
            all_styles.update(styles)
    for style in all_styles:
        feature_list.append(f"style:{style}")

    # Process materials (which is an array in PostgreSQL)   
    all_materials = set()
    for materials in items_df['materials']:
        if isinstance(materials, list):
            all_materials.update(materials)
    for material in all_materials:
        feature_list.append(f"material:{material}")

    # Process sex
    for sex in items_df['sex'].unique():
        if pd.notna(sex):
            feature_list.append(f"sex:{sex}")

    # Process condition
    for condition in items_df['condition'].unique():
        if pd.notna(condition):
            feature_list.append(f"condition:{condition}")

    # Process sizes
    for size in items_df['top_size'].dropna().unique():
        feature_list.append(f'top_size:{size}')
    for size in items_df['bottom_size'].dropna().unique():
        feature_list.append(f'bottom_size:{size}')
    for size in items_df['shoe_size'].dropna().unique():
        feature_list.append(f'shoe_size:{size}')
    
     # Process fit
    for fit in items_df['fit'].dropna().unique():
        feature_list.append(f'fit:{fit}')
        
    # Process price ranges
    items_df['price_range'] = pd.cut(
        items_df['price'], 
        bins=[0, 20, 50, 100, 200, float('inf')],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )
    for price_range in items_df['price_range'].unique():
        if pd.notna(price_range):
            feature_list.append(f'price_range:{price_range}')
        
    # Process season (which is an array in PostgreSQL)
    all_seasons = set()
    for seasons in items_df['season']:
        if isinstance(seasons, list):
            all_seasons.update(seasons)
    for season in all_seasons:
        feature_list.append(f'season:{season}')
        
    # Create item feature matrix
    item_features = []
    for _, item in items_df.iterrows():
        item_id = item['id']
        if item_id not in item_id_map:
            logger.warning(f"Item {item_id} not found in mapping, skipping")
            continue
            
        item_features_dict = {}

        # Add high_category
        if pd.notna(item['high_category']):
            item_features_dict[f'high_category:{item["high_category"]}'] = 1.0
        
        # Add specific_category
        if pd.notna(item['specific_category']):
            item_features_dict[f'specific_category:{item["specific_category"]}'] = 1.0
        
        # Add brand if available
        if pd.notna(item['brand']):
            item_features_dict[f'brand:{item["brand"]}'] = 1.0
        
        # Add colors
        if isinstance(item['colors'], list):
            for color in item['colors']:
                item_features_dict[f'color:{color}'] = 1.0
                
        # Add styles
        if isinstance(item['styles'], list):
            for style in item['styles']:
                item_features_dict[f'style:{style}'] = 1.0
                
        # Add materials
        if isinstance(item['materials'], list):
            for material in item['materials']:
                item_features_dict[f'material:{material}'] = 1.0
                
        # Add sex
        if pd.notna(item['sex']):
            item_features_dict[f'sex:{item["sex"]}'] = 1.0
        
        # Add condition
        if pd.notna(item['condition']):
            item_features_dict[f'condition:{item["condition"]}'] = 1.0
        
        # Add sizes if available
        if pd.notna(item['top_size']):
            item_features_dict[f'top_size:{item["top_size"]}'] = 1.0
        if pd.notna(item['bottom_size']):
            item_features_dict[f'bottom_size:{item["bottom_size"]}'] = 1.0
        if pd.notna(item['shoe_size']):
            item_features_dict[f'shoe_size:{item["shoe_size"]}'] = 1.0
        
        # Add fit if available
        if pd.notna(item['fit']):
            item_features_dict[f'fit:{item["fit"]}'] = 1.0
            
        # Add price range
        if pd.notna(item['price_range']):
            item_features_dict[f'price_range:{item["price_range"]}'] = 1.0
        
        # Add seasons
        if isinstance(item['season'], list):
            for season in item['season']:
                item_features_dict[f'season:{season}'] = 1.0
                
        item_features.append((item_id_map[item_id], item_features_dict))
    
    # Create LightFM dataset with item features
    dataset = Dataset()
    dataset.fit_partial(
        items=list(item_id_map.values()),
        item_features=feature_list
    )

    # Build the item feature matrix
    item_features_matrix = dataset.build_item_features(
        data=item_features,
        normalize=True
    )
    
    logger.info(f"Built item features matrix with {len(item_features)} items")
    return item_features_matrix, dataset

def build_interactions_matrix(interactions_df: pd.DataFrame, user_id_map: Dict[str, int], 
                             item_id_map: Dict[str, int]) -> sparse.coo_matrix:
    """
    Build the interactions matrix directly from the interactions dataframe.
    
    Args:
        interactions_df: DataFrame containing user-item interactions
        user_id_map: Mapping from external user IDs to internal indices
        item_id_map: Mapping from external item IDs to internal indices
        
    Returns:
        Sparse matrix of user-item interactions
    """
    # Handle empty interactions dataframe
    if interactions_df.empty:
        logger.warning("No interactions available. Creating empty interaction matrix.")
        n_users = len(user_id_map) if user_id_map else 1
        n_items = len(item_id_map) if item_id_map else 1
        return sparse.coo_matrix((n_users, n_items), dtype=np.float32)
    
    # Build interactions as (user, item, weight) tuples
    user_ids = []
    item_ids = []
    weights = []
    
    for _, row in interactions_df.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        weight = row['weight']
        
        # Skip if user or item not in mapping
        if user_id not in user_id_map:
            logger.debug(f"User {user_id} not in mapping, skipping")
            continue
            
        if item_id not in item_id_map:
            logger.debug(f"Item {item_id} not in mapping, skipping")
            continue
            
        user_idx = user_id_map[user_id]
        item_idx = item_id_map[item_id]
        
        user_ids.append(user_idx)
        item_ids.append(item_idx)
        weights.append(weight)
    
    # If no valid interactions, return empty matrix
    if not user_ids:
        logger.warning("No valid interactions found. Creating empty interaction matrix.")
        n_users = len(user_id_map) if user_id_map else 1
        n_items = len(item_id_map) if item_id_map else 1
        return sparse.coo_matrix((n_users, n_items), dtype=np.float32)
    
    # Create sparse matrix
    n_users = len(user_id_map)
    n_items = len(item_id_map)
    
    interactions = sparse.coo_matrix(
        (weights, (user_ids, item_ids)),
        shape=(n_users, n_items),
        dtype=np.float32
    )
    
    logger.info(f"Created interaction matrix with {len(weights)} interactions")
    return interactions

def extract_disliked_keywords(supabase, disliked_items: List[str]) -> Tuple[set, set]:
    """
    Extract keywords and categories from disliked items.
    
    Args:
        supabase: Initialized Supabase client
        disliked_items: List of disliked item IDs
        
    Returns:
        Tuple of (disliked_keywords, disliked_categories)
    """
    if not disliked_items:
        return set(), set()
        
    try:
        # Get details of disliked items
        disliked_details = supabase.table('hm_items').select(
            'id, name, high_category, specific_category'
        ).in_('id', disliked_items).execute()
        
        # Extract keywords from disliked items
        disliked_keywords = set()
        disliked_categories = set()
        
        for item in disliked_details.data:
            # Add significant keywords from the name
            name_parts = item['name'].lower().split()
            for part in name_parts:
                if len(part) > 3 and part not in ['with', 'from', 'and', 'the']:
                    disliked_keywords.add(part)
            
            # Add category combinations
            disliked_categories.add((item['high_category'], item['specific_category']))
            
        return disliked_keywords, disliked_categories
    except Exception as e:
        logger.error(f"Error extracting disliked keywords: {e}")
        return set(), set() 