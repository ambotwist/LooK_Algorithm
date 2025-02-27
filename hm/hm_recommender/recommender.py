"""
Main recommender system implementation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from supabase import Client
import scipy.sparse as sparse
import re

from hm_recommender.utils.logger import setup_logger
from hm_recommender.data.fetcher import fetch_items, fetch_interactions, fetch_item_details, fetch_disliked_items
from hm_recommender.data.processor import (
    create_user_item_mappings, 
    process_item_features, 
    build_interactions_matrix,
    extract_disliked_keywords
)
from hm_recommender.models.lightfm_model import LightFMRecommender

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
        
    def _determine_true_category(self, item_details):
        """
        Determine the true category of an item based on its name and other details.
        This helps correct miscategorized items in the database.
        
        Args:
            item_details: Dictionary containing item details
            
        Returns:
            String representing the true high-level category
        """
        if not item_details:
            return None
        
        name = item_details.get('name', '').lower()
        current_high_category = item_details.get('high_category')
        
        # Check for pants/bottoms keywords
        if any(keyword in name for keyword in ['pant', 'chino', 'jean', 'jogger', 'cargo', 'short', 'swim short']):
            return 'bottoms'
        
        # Check for tops keywords
        if any(keyword in name for keyword in ['t-shirt', 'shirt', 'polo', 'tank', 'hoodie', 'sweatshirt', 'sweater']):
            return 'tops'
        
        # Check for outerwear keywords
        if any(keyword in name for keyword in ['jacket', 'coat', 'puffer', 'bomber']):
            return 'outerwear'
        
        # Check for shoes keywords
        if any(keyword in name for keyword in ['shoe', 'sneaker', 'loafer', 'boot', 'sandal']):
            return 'shoes'
        
        # If we couldn't determine a better category, return the current one
        return current_high_category

    def _get_user_category_preferences(self, user_id):
        """
        Analyze user interactions to determine category preferences.
        
        Args:
            user_id: User ID to analyze
            
        Returns:
            Dictionary mapping categories to preference scores
        """
        try:
            # Get user interactions
            response = self.supabase.table('hm_interactions').select('item_id, interaction_type').eq('user_id', user_id).execute()
            
            if not response.data:
                return {}
            
            # Get details of items the user has interacted with
            item_ids = [item['item_id'] for item in response.data]
            items_response = self.supabase.table('hm_items').select('id, name, high_category').in_('id', item_ids).execute()
            
            # Create a mapping of item_id to high_category
            item_categories = {item['id']: self._determine_true_category(item) or item['high_category'] for item in items_response.data}
            
            # Create a mapping of item_id to interaction_type
            item_interactions = {item['item_id']: item['interaction_type'] for item in response.data}
            
            # Calculate category preferences
            category_preferences = {}
            for item_id, interaction_type in item_interactions.items():
                if item_id not in item_categories:
                    continue
                
                category = item_categories[item_id]
                
                # Assign scores based on interaction type
                if interaction_type == 'like':
                    score = 1
                elif interaction_type == 'superlike':
                    score = 3  # Increased from 2 to 3
                elif interaction_type == 'dislike':
                    score = -1
                else:
                    score = 0
                
                category_preferences[category] = category_preferences.get(category, 0) + score
            
            return category_preferences
        except Exception as e:
            self.logger.error(f"Error getting user category preferences: {e}")
            return {}

    def recommend_for_user(self, user_id: str, n: int = 10) -> List[Dict[str, Any]]:
        """
        Generate recommendations for a specific user.
        
        Args:
            user_id: User ID to generate recommendations for
            n: Number of recommendations to generate
            
        Returns:
            List of recommended items
        """
        logger = self.logger
        logger.info(f"Generating recommendations for user {user_id}")
        
        # Get user category preferences
        category_preferences = self._get_user_category_preferences(user_id)
        logger.info(f"User category preferences: {category_preferences}")
        
        # Get items the user has already interacted with
        seen_items = {}
        try:
            response = self.supabase.table('hm_interactions').select('item_id, interaction_type').eq('user_id', user_id).execute()
            
            for item in response.data:
                item_id = item['item_id']
                interaction_type = item['interaction_type']
                seen_items[item_id] = interaction_type
            
            logger.info(f"User has interacted with {len(seen_items)} items")
            
            # Get items the user has liked
            liked_items = [item_id for item_id, interaction_type in seen_items.items()
                          if interaction_type in ['like', 'superlike']]
            
            # Get items the user has disliked
            disliked_items = [item_id for item_id, interaction_type in seen_items.items()
                             if interaction_type == 'dislike']
            
            logger.info(f"User has liked {len(liked_items)} items and disliked {len(disliked_items)} items")
        except Exception as e:
            logger.error(f"Error fetching user interactions: {e}")
            liked_items = []
            disliked_items = []
        
        # If the user has no interactions, use cold start recommendations
        if not seen_items:
            logger.info("User has no interactions, using cold start recommendations")
            return self._cold_start_recommendations(n)
        
        # Get disliked item details to extract keywords for filtering
        disliked_keywords = set()
        try:
            if disliked_items:
                disliked_response = self.supabase.table('hm_items').select('name, description').in_('id', disliked_items).execute()
                
                for item in disliked_response.data:
                    name = item.get('name', '').lower()
                    description = item.get('description', '').lower()
                    
                    # Extract keywords from name and description
                    words = set(re.findall(r'\b\w{3,}\b', name + ' ' + description))
                    disliked_keywords.update(words)
                
                # Remove common words that aren't useful for filtering
                common_words = {'with', 'and', 'the', 'for', 'this', 'that', 'from', 'have', 'has', 'had', 'not', 'are', 'were', 'was', 'been'}
                disliked_keywords = disliked_keywords - common_words
                
                logger.info(f"Extracted {len(disliked_keywords)} keywords from disliked items")
        except Exception as e:
            logger.error(f"Error extracting disliked keywords: {e}")
        
        # Generate recommendations using the trained model
        try:
            # Get all items
            items_response = self.supabase.table('hm_items').select('id').eq('is_active', True).execute()
            all_item_ids = [item['id'] for item in items_response.data]
            
            # Filter out items the user has already interacted with
            candidate_items = [item_id for item_id in all_item_ids if item_id not in seen_items]
            
            logger.info(f"Found {len(candidate_items)} candidate items for recommendations")
            
            # If we have no candidates, return cold start recommendations
            if not candidate_items:
                logger.warning("No candidate items found, using cold start recommendations")
                return self._cold_start_recommendations(n)
            
            # Get scores for all candidate items
            scores = []
            for item_id in candidate_items:
                try:
                    # Get model score
                    score = self.model.predict(user_id, item_id)
                    
                    # Get item details for filtering and boosting
                    item_response = self.supabase.table('hm_items').select('*').eq('id', item_id).execute()
                    if not item_response.data:
                        continue
                    
                    item_details = item_response.data[0]
                    
                    # Determine true category
                    true_category = self._determine_true_category(item_details) or item_details.get('high_category')
                    
                    # Apply category preference boost
                    category_score = category_preferences.get(true_category, 0)
                    category_boost = 0
                    
                    if category_score > 0:
                        # Boost items from categories the user likes
                        category_boost = min(0.5, category_score * 0.1)
                    elif category_score < 0:
                        # Penalize items from categories the user dislikes
                        category_boost = max(-0.5, category_score * 0.1)
                    
                    # Apply the boost
                    score += category_boost
                    
                    scores.append({
                        'item_id': item_id,
                        'score': score,
                        'item_details': item_details,
                        'category_boost': category_boost
                    })
                except Exception as e:
                    logger.error(f"Error getting score for item {item_id}: {e}")
            
            # Sort by score (highest first)
            scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Take top N*2 for filtering
            recommendations = scores[:n*2]
            
            logger.info(f"Generated {len(recommendations)} initial recommendations")
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._cold_start_recommendations(n)
        
        # Apply final filtering based on disliked keywords
        filtered_recommendations = []
        try:
            for rec in recommendations:
                item_details = rec.get('item_details', {})
                name = item_details.get('name', '').lower() if item_details else ''
                description = item_details.get('description', '').lower() if item_details else ''
                
                # Check for disliked keywords
                matching_keywords = []
                keyword_penalty = 0
                
                for keyword in disliked_keywords:
                    if keyword in name or keyword in description:
                        # Apply a penalty based on keyword length (longer keywords are more specific)
                        penalty = min(0.5, len(keyword) / 20)
                        keyword_penalty += penalty
                        matching_keywords.append(keyword)
                
                # Only filter out items with significant keyword penalties
                if keyword_penalty > 0.8:  # Reduced from 1.0 to be more aggressive
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
        
        # Implement category diversity
        try:
            logger.info("Applying category diversity to recommendations")
            diverse_recommendations = []
            seen_categories = set()
            
            # First pass: get one item from each category
            for rec in filtered_recommendations:
                item_details = rec.get('item_details', {})
                name = item_details.get('name', '').lower() if item_details else ''
                
                # Determine the true category based on item name
                true_category = self._determine_true_category(item_details)
                
                if true_category and true_category not in seen_categories:
                    diverse_recommendations.append(rec)
                    seen_categories.add(true_category)
                    logger.debug(f"Added diverse item from category: {true_category}, item: {name}")
                    
                    if len(diverse_recommendations) >= n:
                        break
            
            # Second pass: fill remaining slots with highest scored items
            if len(diverse_recommendations) < n:
                remaining_slots = n - len(diverse_recommendations)
                existing_ids = [rec['item_id'] for rec in diverse_recommendations]
                
                for rec in filtered_recommendations:
                    if rec['item_id'] not in existing_ids:
                        diverse_recommendations.append(rec)
                        logger.debug(f"Added additional item to fill diversity slots: {rec.get('item_id')}")
                        
                        if len(diverse_recommendations) >= n:
                            break
            
            # Sort by score (highest first)
            diverse_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            logger.info(f"Applied category diversity: {len(filtered_recommendations)} → {len(diverse_recommendations)} recommendations")
            
            # If we somehow lost all recommendations, fall back to the filtered ones
            if not diverse_recommendations and filtered_recommendations:
                logger.warning("Diversity filtering removed all recommendations! Falling back to original filtered recommendations.")
                return filtered_recommendations[:n]
                
            return diverse_recommendations[:n]
            
        except Exception as e:
            logger.error(f"Error applying category diversity: {e}")
            # Fall back to filtered recommendations if diversity fails
            logger.info(f"Final recommendations count (without diversity): {len(filtered_recommendations)}")
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
    