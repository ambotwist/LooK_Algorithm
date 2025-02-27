#!/usr/bin/env python3
"""
Script to check the distribution of item categories in the database.
"""

import os
import json
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def main():
    """Main function"""
    # Initialize Supabase client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Check item categories
    print("\n=== Item Categories ===")
    response = supabase.table('hm_items').select('high_category').execute()
    categories = {}
    for item in response.data:
        cat = item['high_category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(json.dumps(categories, indent=2))
    
    # Check specific categories
    print("\n=== Specific Categories ===")
    response = supabase.table('hm_items').select('high_category, specific_category').execute()
    specific_categories = {}
    for item in response.data:
        high_cat = item['high_category']
        specific_cat = item['specific_category']
        key = f"{high_cat}/{specific_cat}"
        specific_categories[key] = specific_categories.get(key, 0) + 1
    
    print(json.dumps(specific_categories, indent=2))
    
    # Check user interactions
    print("\n=== User Interactions ===")
    response = supabase.table('hm_interactions').select('interaction_type').execute()
    interaction_types = {}
    for item in response.data:
        interaction_type = item['interaction_type']
        interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
    
    print(json.dumps(interaction_types, indent=2))
    
    # Check total counts
    print("\n=== Total Counts ===")
    items_response = supabase.table('hm_items').select('*', count='exact').execute()
    interactions_response = supabase.table('hm_interactions').select('*', count='exact').execute()
    
    items_count = items_response.count if hasattr(items_response, 'count') else len(items_response.data)
    interactions_count = interactions_response.count if hasattr(interactions_response, 'count') else len(interactions_response.data)
    
    print(f"Total items: {items_count}")
    print(f"Total interactions: {interactions_count}")
    
    # Check user's interactions
    print("\n=== User's Interactions ===")
    user_id = "e4d98795-62a8-4438-9136-5117ca6e6aee"  # The test user ID
    user_interactions = supabase.table('hm_interactions').select('interaction_type, item_id').eq('user_id', user_id).execute()
    
    user_interaction_types = {}
    for item in user_interactions.data:
        interaction_type = item['interaction_type']
        user_interaction_types[interaction_type] = user_interaction_types.get(interaction_type, 0) + 1
    
    print(f"User {user_id} interactions:")
    print(json.dumps(user_interaction_types, indent=2))
    
    # Get details of items the user has interacted with
    if user_interactions.data:
        print("\n=== Items User Has Interacted With ===")
        item_ids = [item['item_id'] for item in user_interactions.data]
        items_details = supabase.table('hm_items').select('id, name, high_category, specific_category').in_('id', item_ids).execute()
        
        for item in items_details.data:
            interaction_type = next((i['interaction_type'] for i in user_interactions.data if i['item_id'] == item['id']), None)
            print(f"{interaction_type}: {item['name']} ({item['high_category']}/{item['specific_category']})")

if __name__ == "__main__":
    main() 