"""
Command-line interface for H&M item generation.
"""

import argparse
import sys
import time

from .generator import generate_test_items, insert_test_items
from .api import clear_cache
from .utils import clear_database

def main():
    """Main entry point for the H&M item generation script."""
    parser = argparse.ArgumentParser(description='Generate test items from H&M products')
    parser.add_argument('num_items', type=int, nargs='?', default=20,
                        help='Number of items to generate (default: 20)')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh of API cache for the requests made')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear all cached API responses before starting')
    parser.add_argument('--clear-all', action='store_true',
                        help='Clear both the API cache and all items in the database before starting')
    
    args = parser.parse_args()
    
    if args.num_items <= 0:
        print("Number of items must be greater than 0")
        sys.exit(1)
    
    # Handle clearing options
    if args.clear_all:
        # Clear both cache and database
        print("Clearing both cache and database...")
        cache_cleared = clear_cache()
        if not cache_cleared:
            print("Failed to clear cache")
            sys.exit(1)
        
        db_cleared = clear_database()
        if not db_cleared:
            print("Failed to clear database")
            sys.exit(1)
        
        print("Successfully cleared both cache and database")
    elif args.clear_cache:
        # Clear only the cache
        if clear_cache():
            print("Cache cleared successfully")
        else:
            print("Failed to clear cache")
            sys.exit(1)
    
    start_time = time.time()
    
    # Generate test items
    items = generate_test_items(args.num_items, args.force_refresh)
    
    if not items:
        print("No items were generated. Exiting.")
        sys.exit(1)
    
    # Insert test items into Supabase
    insert_test_items(items)
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nGenerated and inserted {len(items)} items in {duration:.2f} seconds")
    print(f"Average time per item: {duration / len(items):.2f} seconds")

if __name__ == '__main__':
    main() 