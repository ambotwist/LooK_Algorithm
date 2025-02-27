"""
API-related functions for H&M item generation.
"""

import os
import http.client
import json
import time
import hashlib

# Get API key from environment
HM_API_KEY = os.getenv("HM_API_KEY")

def get_hm_data(endpoint, params=""):
    """Make a request to the H&M API"""
    print(f"\nMaking API request to: {endpoint} with params: {params}")
    conn = http.client.HTTPSConnection("apidojo-hm-hennes-mauritz-v1.p.rapidapi.com")
    
    headers = {
        'x-rapidapi-key': HM_API_KEY,
        'x-rapidapi-host': "apidojo-hm-hennes-mauritz-v1.p.rapidapi.com"
    }
    
    try:
        conn.request("GET", f"/{endpoint}?{params}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))
        print(f"Response status: {res.status}")
        print(f"Response data preview: {str(response_data)[:200]}...")
        return response_data
    except Exception as e:
        print(f"Error in API request: {str(e)}")
        return {}

def get_hm_data_with_cache(endpoint, params="", cache_dir="api_cache", force_refresh=False):
    """Make a request to the H&M API with caching"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache key based on endpoint and params
    cache_key = hashlib.md5(f"{endpoint}?{params}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
    # If force_refresh is True, remove the specific cache file if it exists
    if force_refresh and os.path.exists(cache_file):
        try:
            os.remove(cache_file)
            print(f"Removed cache file for: {endpoint} with params: {params}")
        except Exception as e:
            print(f"Error removing cache file: {str(e)}")
    
    # Check if cache file exists and is not too old (e.g., less than 24 hours)
    if os.path.exists(cache_file):
        file_age = time.time() - os.path.getmtime(cache_file)
        if file_age < 86400:  # 24 hours in seconds
            try:
                with open(cache_file, 'r') as f:
                    print(f"Using cached response for: {endpoint} with params: {params}")
                    return json.load(f)
            except Exception as e:
                print(f"Error reading cache: {str(e)}")
    
    # Make the API request
    response_data = get_hm_data(endpoint, params)
    
    # Cache the response if it's valid
    if response_data:
        try:
            with open(cache_file, 'w') as f:
                json.dump(response_data, f)
        except Exception as e:
            print(f"Error writing cache: {str(e)}")
    
    return response_data

def get_product_details(product_code):
    """Get detailed information about a specific product"""
    return get_hm_data("products/detail", f"lang=en&country=us&productcode={product_code}")

def get_product_details_with_cache(product_code, force_refresh=False):
    """Get detailed information about a specific product with caching"""
    return get_hm_data_with_cache("products/detail", f"lang=en&country=us&productcode={product_code}", force_refresh=force_refresh)

def fetch_all_products(num_items_needed, pagesize=100, max_pages=10, force_refresh=False):
    """Fetch products from multiple pages until we have enough unique items"""
    all_products = []
    page = 1
    total_count = 0
    
    while True:
        print(f"\nFetching page {page}...")
        products_data = get_hm_data_with_cache("products/list", 
            f"country=us&lang=en&currentpage={page}&pagesize={pagesize}&categories=men_all",
            force_refresh=force_refresh
        )
        
        if not products_data:
            print(f"No data returned for page {page}")
            break
            
        # Get total count from first page
        if page == 1:
            total_count = products_data.get("totalRecords", 0)
            print(f"Total available products: {total_count}")
            
        products = products_data.get("results", [])
        if not products:
            print(f"No more products found on page {page}")
            break
            
        all_products.extend(products)
        print(f"Found {len(products)} products on page {page}")
        print(f"Total products collected so far: {len(all_products)}")
        
        # Check if we've collected enough products (with buffer for duplicates)
        if len(all_products) >= min(total_count, num_items_needed * 2):
            print("Collected enough products for processing")
            break
        
        # Check if we've reached the last page or max pages limit
        if len(products) < pagesize or len(all_products) >= total_count or page >= max_pages:
            print("Reached last page or max pages limit")
            break
        
        page += 1
        time.sleep(1)  # Add delay to avoid rate limiting
    
    print(f"\nTotal products fetched: {len(all_products)} out of {total_count} available")
    return all_products

def clear_cache(cache_dir="api_cache"):
    """Clear all cached API responses"""
    if os.path.exists(cache_dir):
        try:
            # List all files in the cache directory
            cache_files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.endswith('.json')]
            
            # Remove each file
            for file_path in cache_files:
                os.remove(file_path)
                
            print(f"Cleared {len(cache_files)} cache files from {cache_dir}")
            return True
        except Exception as e:
            print(f"Error clearing cache: {str(e)}")
            return False
    else:
        print(f"Cache directory {cache_dir} does not exist")
        return True  # Return True since there's no cache to clear 