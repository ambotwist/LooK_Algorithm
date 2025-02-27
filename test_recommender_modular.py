#!/usr/bin/env python3
"""
Test script for the modular H&M recommender system.
"""

import sys
from hm_recommender import HMRecommender, init_supabase
from typing import List, Dict, Any

def test_recommender():
    """
    Test the H&M recommender system with proper error handling.
    """
    try:
        # Initialize Supabase client
        supabase = init_supabase()
        
        print("Checking database tables...")
        # Check if the interactions table exists and has the right structure
        interactions_response = supabase.table('hm_interactions').select('*').execute()
        
        # Check if interactions table is empty
        if not interactions_response.data:
            print("\nERROR: No interactions found in the database.")
            print("The recommender system requires user interactions to provide personalized recommendations.")
            print("Please add some interactions to the hm_interactions table before running this script.")
            sys.exit(1)
            
        print(f"Interactions table exists with {len(interactions_response.data)} interactions")
        print(f"Sample interaction columns: {list(interactions_response.data[0].keys()) if interactions_response.data else 'No data'}")
        
        # Check if items table exists and has data
        items_response = supabase.table('hm_items').select('id').execute()
        if not items_response.data:
            print("\nERROR: No items found in the database.")
            print("The recommender system requires items to provide recommendations.")
            print("Please add some items to the hm_items table before running this script.")
            sys.exit(1)
            
        print(f"Items table exists with {len(items_response.data)} items")
        
        # Initialize recommender
        recommender = HMRecommender(supabase)
        
        print("\nTraining model...")
        # Train model
        recommender.train_model(
            num_components=30,
            learning_rate=0.05,
            epochs=20,
            loss='warp-kos'
        )
        
        # Get user ID from command line or use default
        if len(sys.argv) > 1:
            user_id = sys.argv[1]
            print(f"\nGenerating recommendations for user: {user_id}")
            recommendations = recommender.recommend_for_user(user_id, n=10)
            
            # Print recommendations
            print(f"\nTop 10 recommendations for user {user_id}:")
            
            # Debug: Print raw recommendations
            print(f"Debug - Number of recommendations: {len(recommendations)}")
            if recommendations:
                print(f"Debug - First recommendation: {recommendations[0]}")
            
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
        else:
            print("\nNo user ID provided. Please run with a user ID: python test_recommender_modular.py <user_id>")
            
    except Exception as e:
        print(f"\nError testing recommender: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_recommender() 