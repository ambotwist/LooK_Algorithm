"""
H&M Item Generation Package

This package contains modules for generating H&M test items for the LooK database.
"""

from .api import get_hm_data, get_hm_data_with_cache, get_product_details, get_product_details_with_cache
from .mapping import map_hm_category, map_hm_color, map_hm_specific_category, map_style
from .utils import get_existing_items, is_duplicate_item, determine_seasons, validate_size
from .generator import generate_test_items, insert_test_items

__all__ = [
    'get_hm_data', 
    'get_hm_data_with_cache', 
    'get_product_details', 
    'get_product_details_with_cache',
    'map_hm_category', 
    'map_hm_color', 
    'map_hm_specific_category', 
    'map_style',
    'get_existing_items', 
    'is_duplicate_item', 
    'determine_seasons', 
    'validate_size',
    'generate_test_items', 
    'insert_test_items'
] 