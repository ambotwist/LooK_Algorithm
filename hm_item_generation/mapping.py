"""
Mapping functions for H&M item generation.
"""

# Constants
VALID_STYLES = ['casual', 'formal', 'sporty', 'vintage', 'streetwear', 'bohemian', 'minimalist', 'preppy', 'punk', 'business', 'chic']
VALID_TOP_SIZES = ["xs", "s", "m", "l", "xl", "xxl"]

def map_hm_category(category_name, product_name):
    """Map H&M categories to our database categories"""
    name_lower = product_name.lower()
    
    # First try to map based on product name - order matters here!
    # Check for bottoms first (pants, shorts, etc.)
    if any(word in name_lower for word in ["pant", "chino", "jean", "jogger", "cargo", "short", "swim short", "skirt", "legging"]):
        return "bottoms"
    # Then check for shoes
    elif any(word in name_lower for word in ["shoe", "sneaker", "boot", "sandal", "loafer"]):
        return "shoes"
    # Then check for outerwear
    elif any(word in name_lower for word in ["jacket", "coat", "bomber"]):
        return "outerwear"
    # Then check for formal wear
    elif any(word in name_lower for word in ["suit", "tuxedo", "blazer"]):
        return "formal_wear"
    # Finally check for tops
    elif any(word in name_lower for word in ["t-shirt", "shirt", "sweater", "hoodie", "top", "polo", "tank", "sweatshirt"]):
        return "tops"
    # Skip accessories (already handled elsewhere)
    
    # Fallback to category mapping
    category_mapping = {
        "Shirts": "tops",
        "T-shirts & Tanks": "tops",
        "Hoodies & Sweatshirts": "tops",
        "Pants": "bottoms",
        "Jeans": "bottoms",
        "Shorts": "bottoms",
        "Jackets & Coats": "outerwear",
        "Blazers & Suits": "formal_wear",
        "Shoes": "shoes",
        "Sportswear": "sportswear",
        "Accessories": "accessories",
        "Socks": "accessories",
        "Bags": "accessories",
        "Belts": "accessories",
        "Hats": "accessories",
        "Jewelry": "accessories"
    }
    return category_mapping.get(category_name, "tops")

def map_hm_color(color_text):
    """Map H&M colors to our database colors"""
    # If no color text is provided, return unknown
    if not color_text:
        return "unknown"
        
    # Expanded color mapping with more variations
    color_mapping = {
        # Basic colors
        "black": "black",
        "white": "white",
        "red": "red",
        "blue": "blue",
        "green": "green",
        "yellow": "yellow",
        "purple": "purple",
        "pink": "pink",
        "orange": "orange",
        "brown": "brown",
        "gray": "grey",
        "grey": "grey",
        
        # Extended variations
        "navy": "blue",
        "light blue": "blue",
        "dark blue": "blue",
        "turquoise": "blue",
        "teal": "blue",
        "aqua": "blue",
        "azure": "blue",
        "cobalt": "blue",
        "indigo": "blue",
        
        "burgundy": "red",
        "maroon": "red",
        "crimson": "red",
        "scarlet": "red",
        "wine": "red",
        "ruby": "red",
        
        "olive": "green",
        "lime": "green",
        "emerald": "green",
        "mint": "green",
        "sage": "green",
        "forest": "green",
        "khaki": "green",
        
        "gold": "yellow",
        "mustard": "yellow",
        "amber": "yellow",
        "lemon": "yellow",
        
        "lavender": "purple",
        "violet": "purple",
        "plum": "purple",
        "lilac": "purple",
        "mauve": "purple",
        
        "rose": "pink",
        "fuchsia": "pink",
        "magenta": "pink",
        "salmon": "pink",
        "coral": "pink",
        
        "rust": "orange",
        "tangerine": "orange",
        "peach": "orange",
        
        "tan": "brown",
        "chocolate": "brown",
        "coffee": "brown",
        "caramel": "brown",
        "beige": "brown",
        
        "silver": "grey",
        "charcoal": "grey",
        "slate": "grey",
        
        "cream": "white",
        "ivory": "white",
        "off-white": "white"
    }
    
    # Convert color text to lowercase for case-insensitive matching
    color_lower = color_text.lower()
    
    # Check for compound colors or patterns
    if any(term in color_lower for term in ["multi", "pattern", "print", "stripe", "check", "floral", "&"]):
        return "multi"
        
    # Try direct mapping first
    if color_lower in color_mapping:
        return color_mapping[color_lower]
        
    # Try partial matching
    for key, value in color_mapping.items():
        if key in color_lower:
            return value
            
    # If we still can't find a match, return unknown
    return "unknown"

def map_hm_specific_category(high_category, product_name):
    """Map H&M categories to specific categories based on high category"""
    name_lower = product_name.lower()
    
    # Category-specific mappings
    category_mappings = {
        "tops": {
            "t-shirt": "t-shirts",
            "shirt": "shirts",
            "tank": "tank_tops",
            "sweater": "sweaters",
            "hoodie": "hoodies",
            "blouse": "blouses",
            "polo": "polo_shirts",
            "sweatshirt": "hoodies"
        },
        "bottoms": {
            "jean": "jeans",
            "short": "shorts",
            "swim short": "shorts",
            "skirt": "skirts",
            "legging": "leggings",
            "pant": "pants",
            "chino": "pants",
            "jogger": "pants",
            "cargo": "pants"
        },
        "shoes": {
            "sneaker": "sneakers",
            "boot": "boots",
            "sandal": "sandals",
            "flat": "flats",
            "heel": "heels",
            "loafer": "loafers"
        },
        "outerwear": {
            "jacket": "jackets",
            "coat": "coats",
            "blazer": "blazers",
            "vest": "vests",
            "bomber": "jackets"
        },
        "formal_wear": {
            "suit": "suits",
            "tuxedo": "tuxedos",
            "dress": "dresses"
        },
        "accessories": {
            "belt": "belts",
            "sock": "socks",
            "bag": "bags",
            "hat": "hats",
            "cap": "hats",
            "scarf": "scarves",
            "glove": "gloves",
            "wallet": "wallets",
            "watch": "watches",
            "sunglasses": "eyewear",
            "jewelry": "jewelry",
            "necklace": "jewelry",
            "bracelet": "jewelry",
            "earring": "jewelry",
            "ring": "jewelry",
            "tie": "ties",
            "backpack": "bags",
            "crossbody": "bags",
            "tote": "bags"
        }
    }
    
    # Get the mapping for the high category
    mapping = category_mappings.get(high_category, {})
    
    # Try to find a match in the product name
    for key, value in mapping.items():
        if key in name_lower:
            return value
            
    # Default values for each high category
    defaults = {
        "tops": "t-shirts",
        "bottoms": "pants",
        "shoes": "sneakers",
        "outerwear": "jackets",
        "formal_wear": "suits",
        "sportswear": "gym_tops",
        "accessories": "other_accessories"
    }
    
    return defaults.get(high_category, "t-shirts")

def map_style(fit):
    """Map H&M fit to valid style"""
    style_mapping = {
        "Slim fit": "minimalist",
        "Regular fit": "casual",
        "Loose fit": "streetwear",
        "Relaxed fit": "casual",
        "Oversized": "streetwear",
        "Skinny fit": "minimalist"
    }
    return style_mapping.get(fit, "casual") 