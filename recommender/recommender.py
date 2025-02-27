"""
Main recommender system implementation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from supabase import Client
import scipy.sparse as sparse

from recommender.utils.logger import setup_logger
from recommender.data.fetcher import fetch_items, fetch_interactions, fetch_item_details, fetch_disliked_items
from recommender.data.processor import (
    create_user_item_mappings, 
    process_item_features, 
    build_interactions_matrix,
    extract_disliked_keywords
)
from recommender.models.lightfm_model import LightFMRecommender

logger = setup_logger("hm_recommender.main")

class HMRecommender:
    """
    A recommendation system for H&M clothing items.

    This class handles:
    1. Loading data from Supabase
    2. Processing item features and user interactions
    3. Training a hybrid recommendation model
    4. Generating personalized recommendations
    5. Evaluating model performance
    """

    def __init__(self, supabase_client: Client, random_state: int = 42):
        """
        Initialize the recommender system.

        Args:
            supabase_client: Supabase client for database access
            random_state: Seed for random number generation (for reproducibility)
        """
        self.supabase = supabase_client
        self.random_state = random_state
        self.model = LightFMRecommender(random_state=random_state)
        self.item_features = None
        self.user_id_map = {}  # Maps external user IDs to internal indices
        self.item_id_map = {}  # Maps external item IDs to internal indices
        
    def train_model(self, num_components: int = 30, learning_rate: float = 0.05, 
                   epochs: int = 20, loss: str = 'warp') -> None:
        """
        Train the recommendation model.
        
        Args:
            num_components: Number of latent factors (embeddings dimension)
            learning_rate: Learning rate for the training algorithm
            epochs: Number of training epochs
            loss: Loss function ('warp', 'bpr', 'logistic', or 'warp-kos')
                  - 'warp': Weighted Approximate-Rank Pairwise (recommended for implicit feedback)
                  - 'bpr': Bayesian Personalized Ranking (for implicit feedback)
                  - 'logistic': Logistic loss (for explicit feedback)
                  - 'warp-kos': k-Order Statistic loss (for diversity)
                  
        Raises:
            ValueError: If no interactions are available for training
        """
        # Fetch data
        items_df = fetch_items(self.supabase)
        interactions_df = fetch_interactions(self.supabase)
        
        # Create mappings
        self.user_id_map, self.item_id_map = create_user_item_mappings(interactions_df, items_df)
        
        # Process item features
        self.item_features, _ = process_item_features(items_df, self.item_id_map)
        
        # Build interaction matrix
        interaction_matrix = build_interactions_matrix(interactions_df, self.user_id_map, self.item_id_map)
        
        # Train the model
        self.model.train(
            interactions=interaction_matrix,
            item_features=self.item_features,
            num_components=num_components,
            learning_rate=learning_rate,
            epochs=epochs,
            loss=loss
        )
        
    def recommend_for_user(self, user_id: str, n: int = 10, exclude_seen: bool = True) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: ID of the user to generate recommendations for
            n: Number of recommendations to generate
            exclude_seen: Whether to exclude items the user has already interacted with
            
        Returns:
            List of recommended items with scores
        """
        # Check if user exists in the model
        if user_id not in self.user_id_map:
            logger.warning(f"User {user_id} not found in training data. Using cold start strategy.")
            return self._cold_start_recommendations(n)
        
        # Get internal user ID
        user_idx = self.user_id_map[user_id]
        logger.info(f"Found user {user_id} with internal index {user_idx}")

        # Get all item scores
        scores = self.model.predict(
            user_ids=user_idx,
            item_ids=np.arange(len(self.item_id_map)),
            item_features=self.item_features
        )
        logger.info(f"Generated scores for {len(scores)} items")
        logger.info(f"Score range: min={np.min(scores)}, max={np.max(scores)}")

        # Handle user interactions
        try:
            # Get disliked items
            disliked_items = fetch_disliked_items(self.supabase, user_id)
            logger.info(f"Found {len(disliked_items)} disliked items for user {user_id}")
            
            # Convert external IDs to internal indices
            disliked_indices = [self.item_id_map[item_id] for item_id in disliked_items if item_id in self.item_id_map]
            logger.info(f"Converted {len(disliked_indices)} disliked items to internal indices")
            
            # Set scores of disliked items to very negative values
            for idx in disliked_indices:
                scores[idx] = -np.inf
                
            # If exclude_seen is True, also exclude liked items
            if exclude_seen:
                # Fetch all interactions
                response = self.supabase.table('hm_interactions').select('item_id, interaction_type').eq('user_id', user_id).execute()
                logger.info(f"Found {len(response.data)} interactions for user {user_id}")
                
                # Create dictionaries for different interaction types
                seen_items = {}
                for item in response.data:
                    item_id = item['item_id']
                    interaction_type = item['interaction_type']
                    seen_items[item_id] = interaction_type
                
                # Get liked items
                liked_items = [item_id for item_id, interaction_type in seen_items.items() 
                              if interaction_type in ['like', 'superlike']]
                liked_indices = [self.item_id_map[item_id] for item_id in liked_items if item_id in self.item_id_map]
                logger.info(f"Found {len(liked_items)} liked items, converted {len(liked_indices)} to internal indices")
                
                # Set scores of liked items to very negative values
                for idx in liked_indices:
                    scores[idx] = -np.inf
                    
            # Extract keywords from disliked items
            if disliked_items:
                disliked_keywords, disliked_categories = extract_disliked_keywords(self.supabase, disliked_items)
                logger.info(f"Extracted {len(disliked_keywords)} keywords and {len(disliked_categories)} categories from disliked items")
                
                # Get all items to check for similar items
                all_items = self.supabase.table('hm_items').select(
                    'id, name, high_category, specific_category'
                ).execute()
                
                # Find items similar to disliked items
                for item in all_items.data:
                    item_id = item['id']
                    if item_id in self.item_id_map and item_id not in disliked_items:
                        # Check if item has similar keywords
                        name_lower = item['name'].lower()
                        
                        # Check for keyword matches
                        keyword_match = False
                        for keyword in disliked_keywords:
                            if keyword in name_lower and len(keyword) > 3:  # Only consider significant keywords
                                keyword_match = True
                                break
                        
                        # Check for category matches
                        category_key = (item['high_category'], item['specific_category'])
                        category_match = category_key in disliked_categories
                        
                        # If there's a strong match, reduce the score
                        if keyword_match:
                            idx = self.item_id_map[item_id]
                            # Penalize but don't completely exclude
                            scores[idx] -= 1.0
                            
        except Exception as e:
            logger.error(f"Error processing user interactions: {e}")

        # Get top N item indices
        top_item_indices = np.argsort(-scores)[:n*2]  # Get more items than needed in case we filter some out
        logger.info(f"Selected {len(top_item_indices)} top item indices")

        # Convert back to external IDs and create result
        recommendations = []
        
        # Create reverse mapping from internal indices to external IDs
        reverse_item_map = {v: k for k, v in self.item_id_map.items()}
        logger.info(f"Created reverse mapping with {len(reverse_item_map)} items")

        valid_count = 0
        for idx in top_item_indices:
            if idx in reverse_item_map:
                item_id = reverse_item_map[idx]
                if scores[idx] > -np.inf:  # Only include items with valid scores
                    recommendations.append({
                        'item_id': item_id,
                        'score': float(scores[idx])
                    })
                    valid_count += 1
                else:
                    logger.debug(f"Skipping item {item_id} with score -inf")
            else:
                logger.warning(f"Item index {idx} not found in reverse mapping")
        
        logger.info(f"Created {valid_count} valid recommendations out of {len(top_item_indices)} top items")
        
        # Add item details to recommendations
        self._add_item_details_to_recommendations(recommendations)
        logger.info(f"Added item details to {len(recommendations)} recommendations")
        
        # Final filtering to remove any remaining items that might match disliked patterns
        filtered_recommendations = []
        try:
            # Initialize disliked_keywords if it doesn't exist
            if 'disliked_keywords' not in locals():
                disliked_keywords = []
                logger.info("No disliked keywords defined, skipping keyword filtering")
            
            # Use a scoring approach instead of binary filtering
            for rec in recommendations:
                item_details = rec.get('item_details', {})
                name = item_details.get('name', '').lower() if item_details else ''
                
                # Calculate a keyword penalty score
                keyword_penalty = 0
                matching_keywords = []
                
                # Only check keywords if we have them and the name
                if disliked_keywords and name:
                    for keyword in disliked_keywords:
                        # Only consider keywords of sufficient length
                        if len(keyword) > 3 and keyword in name:
                            # Longer keywords are more specific and should have higher penalty
                            penalty = min(0.5, len(keyword) / 20)
                            keyword_penalty += penalty
                            matching_keywords.append(keyword)
                
                # Only filter out items with significant keyword penalties
                if keyword_penalty > 1.0:
                    logger.debug(f"Filtering out item '{name}' with penalty {keyword_penalty} due to keywords: {matching_keywords}")
                    continue
                
                # If there's a penalty but not enough to filter out, adjust the score
                if keyword_penalty > 0:
                    rec['score'] -= keyword_penalty
                    logger.debug(f"Applied penalty of {keyword_penalty} to item '{name}' due to keywords: {matching_keywords}")
                    
                filtered_recommendations.append(rec)
                
                if len(filtered_recommendations) >= n:
                    break
                    
            logger.info(f"Applied keyword filtering: {len(recommendations) - len(filtered_recommendations)} items filtered out, {len(filtered_recommendations)} remaining")
            
            # If we filtered everything, return the original recommendations
            if not filtered_recommendations and recommendations:
                logger.warning("All recommendations were filtered out! Returning original recommendations instead.")
                filtered_recommendations = recommendations[:n]
                
        except Exception as e:
            logger.error(f"Error in final filtering: {e}")
            filtered_recommendations = recommendations[:n]  # Fallback to unfiltered recommendations
        
        # Sort by score (highest first)
        filtered_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info(f"Final recommendations count: {len(filtered_recommendations)}")
        return filtered_recommendations[:n]
    
    def _cold_start_recommendations(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate recommendations for new users (cold start).
        
        This uses a simple popularity-based approach combined with diversity.
        
        Args:
            n: Number of recommendations to generate
            
        Returns:
            List of recommended items
        """
        try:
            # Get all items with their interaction counts
            items_response = self.supabase.table('hm_items').select('id, high_category, specific_category').eq('is_active', True).execute()
            items_df = pd.DataFrame(items_response.data)
            
            # Get all interactions
            interactions_response = self.supabase.table('hm_interactions').select('item_id, interaction_type').execute()
            interactions_df = pd.DataFrame(interactions_response.data)
            
            # Count interactions per item
            if not interactions_df.empty:
                # Only count positive interactions (likes and superlikes)
                positive_interactions = interactions_df[interactions_df['interaction_type'].isin(['like', 'superlike'])]
                item_counts = positive_interactions['item_id'].value_counts().to_dict()
                
                # Add interaction counts to items dataframe
                items_df['interaction_count'] = items_df['id'].map(item_counts).fillna(0)
            else:
                items_df['interaction_count'] = 0
                
            # Sort by interaction count
            items_df = items_df.sort_values('interaction_count', ascending=False)
            
            # Get diverse recommendations (one per category combination)
            diverse_items = []
            seen_categories = set()
            
            for _, item in items_df.iterrows():
                category_key = (item['high_category'], item['specific_category'])
                if category_key not in seen_categories:
                    diverse_items.append(item)
                    seen_categories.add(category_key)
                    
                if len(diverse_items) >= n:
                    break
                    
            # If we don't have enough diverse items, add more popular items
            if len(diverse_items) < n:
                remaining = n - len(diverse_items)
                existing_ids = [item['id'] for item in diverse_items]
                
                additional_items = items_df[~items_df['id'].isin(existing_ids)].head(remaining)
                diverse_items.extend(additional_items.to_dict('records'))
                
            # Format recommendations
            recommendations = [{'item_id': item['id'], 'score': 0.0} for item in diverse_items]
            
            # Add item details
            self._add_item_details_to_recommendations(recommendations)
                
            return recommendations
        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {e}")
            return []
    
    def _add_item_details_to_recommendations(self, recommendations: List[Dict[str, Any]]) -> None:
        """
        Add item details to recommendation objects.
        
        Args:
            recommendations: List of recommendation dictionaries with item_id and score
        """
        if not recommendations:
            return
            
        # Get item IDs
        item_ids = [rec['item_id'] for rec in recommendations]
        
        # Fetch item details
        items = fetch_item_details(self.supabase, item_ids)
            
        # Add details to recommendations
        for rec in recommendations:
            item_id = rec['item_id']
            if item_id in items:
                rec['item_details'] = items[item_id] 