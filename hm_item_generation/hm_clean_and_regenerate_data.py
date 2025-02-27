#!/usr/bin/env python3
"""
Script to clean the database and regenerate test data with correct categories.
"""

import os
import sys
from supabase import create_client, Client
from dotenv import load_dotenv
import subprocess
import time

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def init_supabase():
    """Initialize Supabase client"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        sys.exit(1)
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def delete_all_interactions(supabase):
    """Delete all interactions from the hm_interactions table"""
    try:
        print("Counting interactions...")
        response = supabase.table("hm_interactions").select("*", count="exact").execute()
        count = response.count if hasattr(response, 'count') else len(response.data)
        print(f"Found {count} interactions")
        
        if count == 0:
            print("No interactions to delete")
            return
            
        print("Deleting all interactions from hm_interactions table")
        # Use a condition that will match all records
        # For interactions, we'll get all user_ids and delete interactions for each user
        users_response = supabase.table("hm_interactions").select("user_id").execute()
        user_ids = list(set([interaction["user_id"] for interaction in users_response.data]))
        
        print(f"Deleting interactions for {len(user_ids)} users")
        for user_id in user_ids:
            supabase.table("hm_interactions").delete().eq("user_id", user_id).execute()
            
        print("Successfully deleted all interactions")
    except Exception as e:
        print(f"Error deleting interactions: {e}")
        sys.exit(1)

def delete_all_items(supabase):
    """Delete all items from the hm_items table"""
    try:
        print("Counting items...")
        response = supabase.table("hm_items").select("*", count="exact").execute()
        count = response.count if hasattr(response, 'count') else len(response.data)
        print(f"Found {count} items")
        
        if count == 0:
            print("No items to delete")
            return
            
        print("Deleting all items from hm_items table")
        # Get all item IDs and delete them in batches
        items_response = supabase.table("hm_items").select("id").execute()
        item_ids = [item["id"] for item in items_response.data]
        
        batch_size = 50
        for i in range(0, len(item_ids), batch_size):
            batch = item_ids[i:i+batch_size]
            print(f"Deleting batch {i//batch_size + 1}/{(len(item_ids) + batch_size - 1)//batch_size}")
            
            for item_id in batch:
                supabase.table("hm_items").delete().eq("id", item_id).execute()
                
            print(f"Deleted {len(batch)} items")
            time.sleep(1)  # Add a small delay to avoid rate limiting
            
        print("Successfully deleted all items")
    except Exception as e:
        print(f"Error deleting items: {e}")
        sys.exit(1)

def regenerate_data(num_items=100):
    """Regenerate test data using the generate_hm_test_data.py script"""
    try:
        print(f"\nRegenerating {num_items} test items...")
        subprocess.run(["python", "generate_hm_test_data.py", str(num_items)], check=True)
        print("Data regeneration completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error regenerating data: {e}")
        sys.exit(1)

def main():
    """Main function"""
    print("Initializing Supabase client...")
    supabase = init_supabase()
    
    print("\nDeleting all existing interactions...")
    delete_all_interactions(supabase)
    
    print("\nDeleting all existing items...")
    delete_all_items(supabase)
    
    # Ask for confirmation before regenerating
    num_items = input("\nHow many items do you want to generate? (default: 100): ")
    num_items = int(num_items) if num_items.strip() else 100
    
    regenerate_data(num_items)
    
    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main() 