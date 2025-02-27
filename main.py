#!/usr/bin/env python3
"""
Main script to run the H&M recommender system.
"""

import sys
import argparse
from recommender import HMRecommender, init_supabase

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='H&M Recommender System')
    parser.add_argument('--user_id', type=str, help='User ID to generate recommendations for')
    parser.add_argument('--num_recs', type=int, default=10, help='Number of recommendations to generate')
    parser.add_argument('--components', type=int, default=30, help='Number of latent components in the model')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--loss', type=str, default='warp', choices=['warp', 'bpr', 'logistic', 'warp-kos'],
                       help='Loss function to use')
    return parser.parse_args()

def main():
    """Main function to run the recommender system."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Initialize Supabase client
        supabase = init_supabase()
        
        # Initialize recommender
        recommender = HMRecommender(supabase)
        
        # Train model
        print("Training model...")
        recommender.train_model(
            num_components=args.components,
            epochs=args.epochs,
            loss=args.loss
        )
        
        # If no user_id is provided, use a default or prompt for one
        user_id = args.user_id
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
        print(f"\nGenerating recommendations for user: {user_id}")
        recommendations = recommender.recommend_for_user(user_id, n=args.num_recs)
        
        # Print recommendations
        print(f"\nTop {args.num_recs} recommendations for user {user_id}:")
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
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main()) 