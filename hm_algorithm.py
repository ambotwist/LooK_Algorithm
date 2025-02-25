import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score
import scipy.sparse as sparse
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import logging
import os
from typing import List, Dict, Tuple, Optional, Any
from dotenv import load_dotenv
from supabase import create_client, Client


class HMRecommender:
    """
    A recommendation system for H&M clothing items using LightFM.

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
        self.model = None
        self.dataset = None
        self.item_features = None
        self.user_features = None
        self.item_id_map = {} # Maps internal item IDs to external IDs
        self.user_id_map = {} # Maps internal user IDs to external IDs
        self.logger = self.setup_logger()

    def setup_logger(self) -> logging.Logger:
        """Set up a logger for the recommendation system."""
        logger = logging.getLogger("hm_recommender")
        logger.setLevel(logging.INFO)

        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger
    
    def fetch_items(self) -> pd.DataFrame:
        """
        Fetch all clothing items from the hm_items table using Supabase.

        Returns:
            DataFrame containing item data
        """
        try:
            # Fetch items from Supabase
            response = self.supabase.table('hm_items').select(
                'id, brand, price, high_category, specific_category, ' +
                'colors, styles, materials, sex, condition, season, ' +
                'top_size, bottom_size, shoe_size, fit'
            ).eq('is_active', True).execute()
            
            # Convert to DataFrame
            items_df = pd.DataFrame(response.data)
            self.logger.info(f"Fetched {len(items_df)} items from database")
            return items_df
        except Exception as e:
            self.logger.error(f"Error fetching items: {e}")
            raise
        
    def fetch_interactions(self) -> pd.DataFrame:
        """
        Fetch user-item interactions (likes, dislikes, superlikes) using Supabase.
        
        Returns:
            DataFrame containing user interactions
            
        Raises:
            ValueError: If the interactions table doesn't exist or has an unexpected structure
        """
        try:
            # Fetch interactions from Supabase
            response = self.supabase.table('hm_interactions').select(
                'user_id, item_id, interaction_type, created_at'
            ).execute()
            
            # Check if interactions table is empty
            if not response.data:
                error_msg = "No interactions found in the database. The recommender system requires user interactions to provide personalized recommendations."
                self.logger.error(error_msg)
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
                
            self.logger.info(f"Fetched {len(interactions_df)} interactions from database")
            return interactions_df
        except Exception as e:
            self.logger.error(f"Error fetching interactions: {e}")
            # If the table doesn't exist yet, raise a clear error
            if "does not exist" in str(e):
                error_msg = "Interactions table does not exist. Please create the hm_interactions table before running the recommender."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            raise

    def _process_item_features(self, items_df: pd.DataFrame) -> sparse.csr_matrix:
        """
        Process item features into a format suitable for LightFM.
        
        Args:
            items_df: DataFrame containing item data
            
        Returns:
            Sparse matrix of item features
        """
        # Extract and process categorical features
        feature_list = []

        # Process high_category
        for category in items_df['high_category'].unique():
            feature_list.append(f"high_category:{category}")

        # Process specific_category
        for category in items_df['specific_category'].unique():
            feature_list.append(f"specific_category:{category}")
            
        # Process brand
        for brand in items_df['brand'].unique():
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
            feature_list.append(f"sex:{sex}")

        # Process condition
        for condition in items_df['condition'].unique():
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
            item_features_dict = {}

            # Add high_category
            item_features_dict[f'high_category:{item["high_category"]}'] = 1.0
            
            # Add specific_category
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
            item_features_dict[f'sex:{item["sex"]}'] = 1.0
            
            # Add condition
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
            item_features_dict[f'price_range:{item["price_range"]}'] = 1.0
            
            # Add seasons
            if isinstance(item['season'], list):
                for season in item['season']:
                    item_features_dict[f'season:{season}'] = 1.0
                    
            item_features.append((item_id, item_features_dict))
        
        # Create LightFM dataset with item features
        self.dataset = Dataset()
        self.dataset.fit(
            users=[user for user in self.user_id_map.keys()],
            items=[item['id'] for _, item in items_df.iterrows()],
            item_features=feature_list
        )

        # Build the item feature matrix - using the correct method name
        # The correct method is build_item_features, not build_items_mappings
        # We'll create our own item_id_map from the dataset mapping
        mappings = self.dataset.mapping()
        if isinstance(mappings, tuple) and len(mappings) >= 2:
            self.item_id_map = mappings[1]  # Second element is the item mapping
        
        # Create item features in LightFM format
        item_features_matrix = self.dataset.build_item_features(
            data=item_features,
            normalize=True
        )
        
        self.logger.info(f"Processed {len(feature_list)} unique item features")
        return item_features_matrix
    
    def _process_interactions(self, interactions_df: pd.DataFrame) -> sparse.coo_matrix:
        """
        Process user-item interactions into a format suitable for LightFM.
        
        Args:
            interactions_df: DataFrame containing user-item interactions
            
        Returns:
            Sparse matrix of user-item interactions
        """
        # Handle empty interactions dataframe
        if interactions_df.empty:
            self.logger.warning("No interactions available. Creating empty interaction matrix.")
            n_users = len(self.user_id_map) if self.user_id_map else 1
            n_items = len(self.item_id_map) if self.item_id_map else 1
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
            if user_id not in self.user_id_map or item_id not in self.item_id_map:
                continue
                
            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[item_id]
            
            user_ids.append(user_idx)
            item_ids.append(item_idx)
            weights.append(weight)
        
        # If no valid interactions, return empty matrix
        if not user_ids:
            self.logger.warning("No valid interactions found. Creating empty interaction matrix.")
            n_users = len(self.user_id_map) if self.user_id_map else 1
            n_items = len(self.item_id_map) if self.item_id_map else 1
            return sparse.coo_matrix((n_users, n_items), dtype=np.float32)
        
        # Create sparse matrix
        n_users = max(user_ids) + 1
        n_items = max(item_ids) + 1
        
        interactions = sparse.coo_matrix(
            (weights, (user_ids, item_ids)),
            shape=(n_users, n_items),
            dtype=np.float32
        )
        
        return interactions
    
    def train_model(self, num_components: int = 30, learning_rate: float = 0.05, 
                   epochs: int = 20, loss: str = 'warp') -> None:
        """
        Train the LightFM recommendation model.
        
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
        items_df = self.fetch_items()
        interactions_df = self.fetch_interactions()
        
        # Check if interactions dataframe is empty
        if interactions_df.empty:
            error_msg = "No interactions available for training. The recommender system requires user interactions to provide personalized recommendations."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Initialize dataset
        self.dataset = Dataset()
        
        # First, collect all unique item features for fitting
        feature_list = []
        
        # Process high_category
        for category in items_df['high_category'].unique():
            feature_list.append(f"high_category:{category}")

        # Process specific_category
        for category in items_df['specific_category'].unique():
            feature_list.append(f"specific_category:{category}")
            
        # Process brand
        for brand in items_df['brand'].unique():
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

        # Process other features (simplified for brevity)
        for sex in items_df['sex'].unique():
            feature_list.append(f"sex:{sex}")
        for condition in items_df['condition'].unique():
            feature_list.append(f"condition:{condition}")
        
        # Create our own mappings for users and items
        # This ensures all items are included
        self.user_id_map = {}
        self.item_id_map = {}
        
        # Create user mapping
        for i, user_id in enumerate(interactions_df['user_id'].unique()):
            self.user_id_map[user_id] = i
            
        # Create item mapping for all items
        for i, item_id in enumerate(items_df['id']):
            self.item_id_map[item_id] = i
            
        self.logger.info(f"Created user mapping with {len(self.user_id_map)} users")
        self.logger.info(f"Created item mapping with {len(self.item_id_map)} items")
        
        # Create item features in the format LightFM expects
        item_features = []
        for _, item in items_df.iterrows():
            item_id = item['id']
            if item_id not in self.item_id_map:
                self.logger.warning(f"Item {item_id} not found in mapping, skipping")
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
            
            # Add to feature list
            item_features.append((self.item_id_map[item_id], item_features_dict))
        
        # Fit dataset with item features
        self.dataset.fit_partial(
            items=list(self.item_id_map.values()),
            item_features=feature_list
        )
        
        # Build the item feature matrix
        self.item_features = self.dataset.build_item_features(
            data=item_features,
            normalize=True
        )
        self.logger.info(f"Built item features matrix with {len(item_features)} items")
        
        # Process interactions
        interaction_matrix = self._build_interactions_matrix(interactions_df)
        
        # Initialize and train the model
        self.model = LightFM(
            no_components=num_components,
            learning_rate=learning_rate,
            loss=loss,
            random_state=self.random_state
        )
        
        self.model.fit(
            interactions=interaction_matrix,
            item_features=self.item_features,
            epochs=epochs,
            verbose=True
        )
        
        self.logger.info(f"Trained model with {num_components} components using {loss} loss")
    
    def _build_interactions_matrix(self, interactions_df: pd.DataFrame) -> sparse.coo_matrix:
        """
        Build the interactions matrix directly from the interactions dataframe.
        
        Args:
            interactions_df: DataFrame containing user-item interactions
            
        Returns:
            Sparse matrix of user-item interactions
        """
        # Handle empty interactions dataframe
        if interactions_df.empty:
            self.logger.warning("No interactions available. Creating empty interaction matrix.")
            n_users = len(self.user_id_map) if self.user_id_map else 1
            n_items = len(self.item_id_map) if self.item_id_map else 1
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
            if user_id not in self.user_id_map:
                self.logger.debug(f"User {user_id} not in mapping, skipping")
                continue
                
            if item_id not in self.item_id_map:
                self.logger.debug(f"Item {item_id} not in mapping, skipping")
                continue
                
            user_idx = self.user_id_map[user_id]
            item_idx = self.item_id_map[item_id]
            
            user_ids.append(user_idx)
            item_ids.append(item_idx)
            weights.append(weight)
        
        # If no valid interactions, return empty matrix
        if not user_ids:
            self.logger.warning("No valid interactions found. Creating empty interaction matrix.")
            n_users = len(self.user_id_map) if self.user_id_map else 1
            n_items = len(self.item_id_map) if self.item_id_map else 1
            return sparse.coo_matrix((n_users, n_items), dtype=np.float32)
        
        # Create sparse matrix
        n_users = len(self.user_id_map)
        n_items = len(self.item_id_map)
        
        interactions = sparse.coo_matrix(
            (weights, (user_ids, item_ids)),
            shape=(n_users, n_items),
            dtype=np.float32
        )
        
        self.logger.info(f"Created interaction matrix with {len(weights)} interactions")
        return interactions
    
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
        if self.model is None:
            self.logger.warning("Model not trained. Call train_model() first.")
            return []
        
        # Check if user exists in the model
        if user_id not in self.user_id_map:
            self.logger.warning(f"User {user_id} not found in training data. Using cold start strategy.")
            return self._cold_start_recommendations(n)
        
        # Get internal user ID
        user_idx = self.user_id_map[user_id]

        # Get all item scores
        scores = self.model.predict(
            user_ids=user_idx,
            item_ids=np.arange(len(self.item_id_map)),
            item_features=self.item_features
        )

        # Get items the user has already interacted with
        if exclude_seen:
            try:
                # Fetch seen items from Supabase
                response = self.supabase.table('hm_interactions').select('item_id').eq('user_id', user_id).execute()
                seen_items = [item['item_id'] for item in response.data]
                
                # Convert external IDs to internal indices
                seen_indices = [self.item_id_map[item_id] for item_id in seen_items if item_id in self.item_id_map]

                # Set scores of seen items to very negative values
                for idx in seen_indices:
                    scores[idx] = -np.inf
            except Exception as e:
                self.logger.error(f"Error fetching seen items: {e}")

        # Get top N item indices
        top_item_indices = np.argsort(-scores)[:n]

        # Convert back to external IDs and create result
        recommendations = []
        
        # Create reverse mapping from internal indices to external IDs
        reverse_item_map = {v: k for k, v in self.item_id_map.items()}

        for idx in top_item_indices:
            if idx in reverse_item_map:
                item_id = reverse_item_map[idx]
                recommendations.append({
                    'item_id': item_id,
                    'score': float(scores[idx])
                })
            else:
                self.logger.warning(f"Item index {idx} not found in reverse mapping")
        
        # Add item details to recommendations
        self._add_item_details_to_recommendations(recommendations)

        return recommendations
    
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
            self.logger.error(f"Error generating cold start recommendations: {e}")
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
        
        try:
            # Fetch item details from Supabase
            # Use 'in' filter for multiple IDs
            response = self.supabase.table('hm_items').select(
                'id, brand, price, high_category, specific_category, colors, styles, name, images'
            ).in_('id', item_ids).execute()
            
            # Convert to dictionary for easy lookup
            items = {item['id']: item for item in response.data}
                
            # Add details to recommendations
            for rec in recommendations:
                item_id = rec['item_id']
                if item_id in items:
                    rec['item_details'] = items[item_id]
        except Exception as e:
            self.logger.error(f"Error fetching item details: {e}")
    
    def evaluate_model(self, test_interactions: pd.DataFrame = None) -> Dict[str, float]:
        """
        Evaluate the model's performance.

        Args:
            test_interactions: Test set of interactions (if None, uses a sample of all interactions)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            self.logger.error("Model not trained. Call train_model() first.")
            return {}
            
        # If no test set provided, use a sample of all interactions
        if test_interactions is None:
            interactions_df = self.fetch_interactions()
            if interactions_df.empty:
                self.logger.warning("No interactions available for evaluation")
                return {}
                
            # Split interactions by timestamp (newer interactions for testing)
            interactions_df = interactions_df.sort_values('timestamp')
            split_idx = int(len(interactions_df) * 0.8)
            train_df = interactions_df.iloc[:split_idx]
            test_df = interactions_df.iloc[split_idx:]
            
            # Process training interactions
            self.dataset = Dataset()
            self.dataset.fit(
                users=train_df['user_id'].unique(),
                items=train_df['item_id'].unique()
            )
            
            # Build interaction matrices
            train_interactions = self._process_interactions(train_df)
            test_interactions = self._process_interactions(test_df)
            
            # Retrain model on training set
            self.model.fit(
                interactions=train_interactions,
                item_features=self.item_features,
                epochs=5,
                verbose=True
            )
        else:
            # Process provided test interactions
            test_interactions = self._process_interactions(test_interactions)
            
        # Calculate metrics
        precision = precision_at_k(
            self.model, 
            test_interactions,
            item_features=self.item_features,
            k=10
        ).mean()
        
        auc = auc_score(
            self.model,
            test_interactions,
            item_features=self.item_features
        ).mean()
        
        metrics = {
            'precision@10': float(precision),
            'auc': float(auc)
        }
        
        self.logger.info(f"Evaluation metrics: {metrics}")
        return metrics


def main(user_id: str = "e4d98795-62a8-4438-9136-5117ca6e6aee"):
    """
    Example usage of the HMRecommender class.
    
    Args:
        user_id: The ID of the user to generate recommendations for
    """
    # Load environment variables
    load_dotenv()
    
    # Initialize Supabase client
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    if not supabase_url or not supabase_key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables must be set.")
        return
        
    supabase = create_client(supabase_url, supabase_key)
    
    # Initialize recommender
    recommender = HMRecommender(supabase)
    
    # Train model
    recommender.train_model(
        num_components=30,
        learning_rate=0.05,
        epochs=20,
        loss='warp'
    )
    
    # If no user_id is provided, use a default or prompt for one
    if user_id is None:
        # Try to get a user ID from the database
        try:
            response = supabase.table('hm_interactions').select('user_id').limit(1).execute()
            if response.data:
                user_id = response.data[0]['user_id']
                print(f"Using user ID from database: {user_id}")
            else:
                print("No user ID provided and none found in database. Please provide a user ID to get recommendations.")
                return
        except Exception as e:
            print(f"Error getting user ID from database: {e}")
            print("Please provide a user ID to get recommendations.")
            return
    
    # Generate recommendations for the specified user
    recommendations = recommender.recommend_for_user(user_id, n=10)
    
    # Print recommendations
    print(f"\nTop 10 recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        item_details = rec.get('item_details', {})
        print(f"{i}. {item_details.get('name', 'Unknown')} - {item_details.get('brand', 'Unknown')}")
        print(f"   Category: {item_details.get('high_category', 'Unknown')} - {item_details.get('specific_category', 'Unknown')}")
        print(f"   Price: ${item_details.get('price', 0):.2f}")
        
        # Display image links if available
        images = item_details.get('images', [])
        if images and isinstance(images, list) and len(images) > 0:
            print(f"   Image: {images[0]}")  # Display the first image link
        
        print(f"   Score: {rec['score']:.4f}")
        print()
    
    # Evaluate model
    metrics = recommender.evaluate_model()
    print(f"Model evaluation metrics: {metrics}")


if __name__ == "__main__":
    import sys
    
    # Check if a user ID was provided as a command-line argument
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
        main(user_id)
    else:
        main()