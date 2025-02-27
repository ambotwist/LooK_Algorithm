"""
Utility functions for H&M item generation.
"""

import re
import os
import time
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Import constants from mapping
from .mapping import VALID_TOP_SIZES

def get_existing_items():
    """Get existing items with their image URLs and names"""
    try:
        response = supabase.table("hm_items").select("external_id, images, name, price").execute()
        existing_items = {
            'codes': set(),
            'images': set(),
            'name_price_pairs': set()  # Tuple of (name, price) to catch variants
        }
        for item in response.data:
            if item.get('external_id'):
                existing_items['codes'].add(item['external_id'])
            if item.get('images'):
                existing_items['images'].update(item['images'])
            if item.get('name') and item.get('price') is not None:
                existing_items['name_price_pairs'].add((item['name'], item['price']))
        return existing_items
    except Exception as e:
        print(f"Error fetching existing items: {str(e)}")
        return {'codes': set(), 'images': set(), 'name_price_pairs': set()}

def clear_database():
    """Delete all items from the hm_items table"""
    try:
        print("Counting items in database...")
        response = supabase.table("hm_items").select("*", count="exact").execute()
        count = response.count if hasattr(response, 'count') else len(response.data)
        print(f"Found {count} items in database")
        
        if count == 0:
            print("No items to delete from database")
            return True
            
        print("Deleting all items from hm_items table...")
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
            
        print("Successfully deleted all items from database")
        return True
    except Exception as e:
        print(f"Error deleting items from database: {e}")
        return False

def is_duplicate_item(product_details, existing_items, images):
    """Check if an item is a duplicate based on various criteria"""
    # Check external_id
    if product_details["code"] in existing_items['codes']:
        print(f"Skipping duplicate product code: {product_details['code']}")
        return True
        
    # Check image URLs
    if any(img in existing_items['images'] for img in images):
        print(f"Skipping product with duplicate image: {product_details['name']}")
        return True
        
    # Check name and price combination
    price = product_details.get("whitePrice", {}).get("price", 0)
    name_price = (product_details['name'], price)
    if name_price in existing_items['name_price_pairs']:
        print(f"Skipping product with duplicate name and price: {product_details['name']}")
        return True
        
    return False

def determine_seasons(product_name, product_details):
    """Determine the appropriate seasons for a product"""
    name_lower = product_name.lower()
    description = product_details.get('description', '').lower()
    text_to_check = f"{name_lower} {description}"
    
    # Define season keywords
    season_keywords = {
        'spring': ['spring', 'lightweight', 'rain'],
        'summer': ['summer', 'beach', 'tropical', 'light', 'breathable'],
        'fall': ['fall', 'autumn', 'transitional'],
        'winter': ['winter', 'warm', 'heavy', 'cold']
    }
    
    # Check for explicit season mentions
    seasons = []
    for season, keywords in season_keywords.items():
        if any(keyword in text_to_check for keyword in keywords):
            seasons.append(season)
    
    # If no seasons found, make educated guess based on materials and product type
    if not seasons:
        materials = [m.lower() for m in product_details.get('keyFibreTypes', [])]
        
        # Summer materials
        if any(m in materials for m in ['linen', 'cotton', 'silk']):
            seasons.extend(['spring', 'summer'])
            
        # Winter materials
        if any(m in materials for m in ['wool', 'fleece', 'down']):
            seasons.extend(['fall', 'winter'])
            
        # Year-round materials with type-based seasonality
        if 'cotton' in materials:
            if any(word in name_lower for word in ['sweater', 'jacket', 'coat']):
                seasons.extend(['fall', 'winter'])
            elif any(word in name_lower for word in ['t-shirt', 'shorts']):
                seasons.extend(['spring', 'summer'])
    
    # Default to all seasons if still no match
    if not seasons:
        if any(word in name_lower for word in ['jacket', 'coat', 'sweater', 'hoodie']):
            seasons = ['fall', 'winter']
        elif any(word in name_lower for word in ['t-shirt', 'tank', 'shorts']):
            seasons = ['spring', 'summer']
        else:
            seasons = ['spring', 'summer', 'fall', 'winter']  # Year-round
    
    return list(set(seasons))  # Remove duplicates

def validate_size(size_type, size_value):
    """Validate size formats"""
    if not size_value:
        return None
    if size_type == "top_size":
        return size_value.lower() if size_value.lower() in VALID_TOP_SIZES else None
    elif size_type == "shoe_size":
        return size_value if re.match(r'^[3-4][0-9]$', str(size_value)) else None
    elif size_type == "bottom_size":
        return size_value if re.match(r'^W[2-4][0-9]L[2-3][0-9]$', str(size_value)) else None
    return None 