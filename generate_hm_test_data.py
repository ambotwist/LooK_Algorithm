import os
import uuid
import http.client
import json
from supabase import create_client, Client
from dotenv import load_dotenv
import re
import time
import argparse
import hashlib

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
HM_API_KEY = os.getenv("HM_API_KEY")  # Add this to your .env file

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Add these constants at the top after imports
VALID_STYLES = ['casual', 'formal', 'sporty', 'vintage', 'streetwear', 'bohemian', 'minimalist', 'preppy', 'punk', 'business', 'chic']
VALID_TOP_SIZES = ["xs", "s", "m", "l", "xl", "xxl"]

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

def get_hm_data_with_cache(endpoint, params="", cache_dir="api_cache"):
    """Make a request to the H&M API with caching"""
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a cache key based on endpoint and params
    cache_key = hashlib.md5(f"{endpoint}?{params}".encode()).hexdigest()
    cache_file = os.path.join(cache_dir, f"{cache_key}.json")
    
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
        "Accessories": "accessories",  # Add this mapping
        "Socks": "accessories",        # Add this mapping
        "Bags": "accessories",         # Add this mapping
        "Belts": "accessories",        # Add this mapping
        "Hats": "accessories",         # Add this mapping
        "Jewelry": "accessories"       # Add this mapping
    }
    return category_mapping.get(category_name, "tops")

def map_hm_color(color_text):
    """Map H&M colors to our database colors"""
    color_mapping = {
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
        "grey": "grey"
    }
    
    # Convert color text to lowercase and try to map
    color_lower = color_text.lower()
    
    # Check for compound colors
    if "multi" in color_lower or "&" in color_lower:
        return "multi"
        
    # Try direct mapping
    for key, value in color_mapping.items():
        if key in color_lower:
            return value
            
    # Default to black if no mapping found
    return "black"

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
        "accessories": "other_accessories"  # Changed from "watches" to more generic
    }
    
    return defaults.get(high_category, "t-shirts")

def get_product_details(product_code):
    """Get detailed information about a specific product"""
    return get_hm_data("products/detail", f"lang=en&country=us&productcode={product_code}")

def get_product_details_with_cache(product_code):
    """Get detailed information about a specific product with caching"""
    return get_hm_data_with_cache("products/detail", f"lang=en&country=us&productcode={product_code}")

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

def map_style(fit):
    """Map H&M fit to valid style"""
    style_mapping = {
        "Slim fit": "minimalist",
        "Regular fit": "casual",
        "Loose fit": "streetwear",
        "Relaxed fit": "casual",
        "Oversized": "streetwear",
        "Skinny fit": "minimalist"  # Add mapping for Skinny fit
    }
    return style_mapping.get(fit, "casual")

def fetch_all_products(num_items_needed, pagesize=100, max_pages=10):
    """Fetch products from multiple pages until we have enough unique items"""
    all_products = []
    page = 1
    total_count = 0
    
    while True:
        print(f"\nFetching page {page}...")
        products_data = get_hm_data_with_cache("products/list", 
            f"country=us&lang=en&currentpage={page}&pagesize={pagesize}&categories=men_all"
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

def generate_test_items(num_items=20):
    """Generate test items from H&M products"""
    print(f"\nStarting to generate {num_items} test items...")
    
    # Get existing items data
    existing_items = get_existing_items()
    print(f"Found {len(existing_items['codes'])} existing H&M items in database")
    
    items = []
    page = 1
    max_pages = 20  # Set a reasonable limit to avoid infinite loops
    
    # Track duplicates within current batch
    current_batch_images = set()
    current_batch_name_price = set()
    
    while len(items) < num_items and page <= max_pages:
        print(f"\nFetching page {page} to find more unique products...")
        
        # Fetch products for the current page
        products_data = get_hm_data_with_cache("products/list", 
            f"country=us&lang=en&currentpage={page}&pagesize=100&categories=men_all"
        )
        
        if not products_data or not products_data.get("results"):
            print("No more products available")
            break
            
        products = products_data.get("results", [])
        print(f"Found {len(products)} products on page {page}")
        
        # Create a mapping of product codes to their corresponding products
        product_map = {}
        for product in products:
            product_code = product["defaultArticle"]["code"]
            product_map[product_code] = product
        
        # Filter out known duplicates by code
        new_product_codes = [code for code in product_map.keys() if code not in existing_items['codes']]
        print(f"Found {len(new_product_codes)} potentially new products on this page")
        
        # Process the filtered products
        for product_code in new_product_codes:
            if len(items) >= num_items:
                break
            
            # Get the basic product info from our mapping
            product = product_map[product_code]
            product_name = product["name"]
            
            # Check for basic duplicates that we can detect without API calls
            # For example, name/price combinations if price is available in the listing
            if "price" in product and product["price"].get("value") is not None:
                price_value = product["price"]["value"]
                name_price = (product_name, price_value)
                if name_price in existing_items['name_price_pairs'] or name_price in current_batch_name_price:
                    print(f"Skipping product with duplicate name and price (pre-check): {product_name}")
                    continue
                
            # Get detailed product information
            details = get_product_details_with_cache(product_code)
            
            if "product" not in details:
                print("No product details found, skipping...")
                continue
                
            product_details = details["product"]
            
            # Map categories
            high_category = map_hm_category(product_details.get("mainCategory", {}).get("name", ""), product_name)
            
            # Skip accessories
            if high_category == "accessories":
                print(f"Skipping accessory item: {product_name}")
                existing_items['codes'].add(product_code)  # Add to existing codes to prevent future API calls
                continue
                
            # Get images
            images = []
            color_image_mapping = {}

            # First check for direct galleryDetails in the product
            if "galleryDetails" in product_details:
                main_color = product_details.get("color", {}).get("text", "")
                main_color_mapped = map_hm_color(main_color)
                main_rgb = product_details.get("color", {}).get("rgbColor", "")
                
                for gallery_item in product_details["galleryDetails"]:
                    if "baseUrl" in gallery_item:
                        image_url = gallery_item["baseUrl"]
                        images.append(image_url)
                        color_image_mapping[image_url] = {
                            "color_name": main_color,
                            "color_mapped": main_color_mapped,
                            "rgb_color": main_rgb
                        }

            # Then check articlesList for color variants
            if "articlesList" in product_details:
                for article in product_details["articlesList"]:
                    variant_color = article.get("color", {}).get("text", "")
                    variant_color_mapped = map_hm_color(variant_color)
                    variant_rgb = article.get("color", {}).get("rgbColor", "")
                    
                    gallery_details = article.get("galleryDetails", [])
                    for gallery_item in gallery_details:
                        if "baseUrl" in gallery_item:
                            image_url = gallery_item["baseUrl"]
                            if image_url not in images:  # Avoid duplicates
                                images.append(image_url)
                                color_image_mapping[image_url] = {
                                    "color_name": variant_color,
                                    "color_mapped": variant_color_mapped,
                                    "rgb_color": variant_rgb
                                }

            # Default image if none found
            if not images:
                # Create placeholder images with different angles/views
                placeholder_urls = [
                    "https://via.placeholder.com/300?text=Front",
                    "https://via.placeholder.com/300?text=Back",
                    "https://via.placeholder.com/300?text=Side"
                ]
                images = placeholder_urls
                for url in placeholder_urls:
                    color_image_mapping[url] = {
                        "color_name": "Default",
                        "color_mapped": "black",
                        "rgb_color": "#000000"
                    }
            # If we have fewer than 3 images, duplicate some to reach minimum
            elif len(images) < 3:
                original_images = images.copy()
                while len(images) < 3:
                    for img in original_images:
                        if len(images) < 3:
                            # Add a slight modification to URL to avoid exact duplicates
                            new_img = f"{img}{'&v=' + str(len(images)) if '?' in img else '?v=' + str(len(images))}"
                            images.append(new_img)
                            color_image_mapping[new_img] = color_image_mapping[img]
                        else:
                            break
            # If we have more than 9 images, keep only the first 9
            elif len(images) > 9:
                # Prioritize different image types if possible
                image_types = {}
                for img in images:
                    for img_type in ["LOOKBOOK", "DESCRIPTIVESTILLLIFE", "DESCRIPTIVEDETAIL"]:
                        if img_type.lower() in img.lower():
                            if img_type not in image_types:
                                image_types[img_type] = []
                            image_types[img_type].append(img)
                
                # Create a balanced selection of images
                selected_images = []
                # First add one of each type if available
                for img_type in ["LOOKBOOK", "DESCRIPTIVESTILLLIFE", "DESCRIPTIVEDETAIL"]:
                    if img_type in image_types and image_types[img_type]:
                        selected_images.append(image_types[img_type][0])
                        image_types[img_type] = image_types[img_type][1:]
                
                # Then fill remaining slots with a balance of available types
                remaining_slots = 9 - len(selected_images)
                if remaining_slots > 0:
                    all_remaining = []
                    for img_list in image_types.values():
                        all_remaining.extend(img_list)
                    all_remaining.extend([img for img in images if img not in selected_images])
                    selected_images.extend(all_remaining[:remaining_slots])
                
                # Update images list and remove unused mappings
                unused_images = [img for img in images if img not in selected_images]
                for img in unused_images:
                    if img in color_image_mapping:
                        del color_image_mapping[img]
                
                images = selected_images[:9]

            print(f"Using {len(images)} images for product {product_name}")
                
            # Check for duplicates and update existing_items if found
            if is_duplicate_item(product_details, existing_items, images):
                # Add the product code to existing codes to prevent future API calls
                existing_items['codes'].add(product_code)
                continue
                
            # Check for duplicates within current batch
            if any(img in current_batch_images for img in images):
                print(f"Skipping product with duplicate image in current batch: {product_details['name']}")
                existing_items['codes'].add(product_code)
                continue
                
            price = product_details.get("whitePrice", {}).get("price", 0)
            name_price = (product_details['name'], price)
            if name_price in current_batch_name_price:
                print(f"Skipping product with duplicate name and price in current batch: {product_details['name']}")
                existing_items['codes'].add(product_code)
                continue
                
            # Update tracking sets
            current_batch_images.update(images)
            current_batch_name_price.add(name_price)
            
            specific_category = map_hm_specific_category(high_category, product_name)
            print(f"Mapped categories: {product_name} -> {high_category} -> {specific_category}")
            
            # Map colors
            colors = []
            if "color" in product_details:
                color = map_hm_color(product_details["color"]["text"])
                colors = [color]
            if not colors:
                colors = ["black"]  # Default color
                
            # Get the primary color name for filtering images
            primary_color_name = product_details.get("color", {}).get("text", "").lower()
            
            # Filter images to prioritize those matching the primary color
            primary_color_images = []
            other_images = []
            
            for img_url in images:
                if img_url in color_image_mapping:
                    img_color_name = color_image_mapping[img_url]["color_name"].lower()
                    if img_color_name == primary_color_name:
                        primary_color_images.append(img_url)
                    else:
                        other_images.append(img_url)
            
            # Use primary color images if we have enough, otherwise supplement with others
            if len(primary_color_images) >= 3:
                # If we have enough primary color images, use only those (up to 9)
                filtered_images = primary_color_images[:9]
            else:
                # If we don't have enough primary color images, use what we have and add others
                # to reach the minimum of 3 (but still cap at 9 total)
                filtered_images = primary_color_images.copy()
                remaining_slots = min(9 - len(filtered_images), len(other_images))
                filtered_images.extend(other_images[:remaining_slots])
                
                # If we still don't have 3 images, duplicate some to reach the minimum
                if len(filtered_images) < 3:
                    original_images = filtered_images.copy()
                    while len(filtered_images) < 3:
                        for img in original_images:
                            if len(filtered_images) < 3:
                                # Add a slight modification to URL to avoid exact duplicates
                                new_img = f"{img}{'&v=' + str(len(filtered_images)) if '?' in img else '?v=' + str(len(filtered_images))}"
                                filtered_images.append(new_img)
                                color_image_mapping[new_img] = color_image_mapping[img]
                            else:
                                break
            
            # Update the images list with our filtered selection
            images = filtered_images
            
            # Update color_image_mapping to remove any unused images
            unused_images = [img for img in color_image_mapping.keys() if img not in images]
            for img in unused_images:
                if img in color_image_mapping:
                    del color_image_mapping[img]
                
            # Determine sizes based on high_category
            top_size = shoe_size = bottom_size = None
            if high_category == "tops":
                top_size = validate_size("top_size", "m")
            elif high_category == "shoes":
                shoe_size = validate_size("shoe_size", "40")
            elif high_category == "bottoms":
                bottom_size = validate_size("bottom_size", "W32L32")
                
            # Extract additional metadata
            description = product_details.get("description", "")
            fits = product_details.get("fits", [])
            style = map_style(fits[0] if fits else None)
            
            # Get measurements and garment details
            measurements = product_details.get("measurements", [])
            length_collection = product_details.get("lengthCollection", [])
            garment_length = None
            waist_rise = None
            for length_item in length_collection:
                if length_item.get("code") == "garmentLength":
                    garment_length = length_item.get("value", [])[0] if length_item.get("value") else None
                elif length_item.get("code") == "waistRise":
                    waist_rise = length_item.get("value", [])[0] if length_item.get("value") else None
                    
            # Get price details
            price_info = product_details.get("whitePrice", {})
            price = price_info.get("price", 0)
            currency = price_info.get("currency", "USD")
            
            # Get RGB color
            rgb_color = product_details.get("color", {}).get("rgbColor")
                
            # Determine seasons
            seasons = determine_seasons(product_name, product_details)
            
            item = {
                "id": str(uuid.uuid4()),
                "external_id": product_code,
                "brand": "H&M",
                "price": price,
                "high_category": high_category,
                "specific_category": specific_category,
                "colors": colors,
                "styles": [style],
                "materials": [material.lower() for material in product_details.get("keyFibreTypes", ["cotton"])],
                "season": seasons,
                "top_size": top_size,
                "shoe_size": shoe_size,
                "bottom_size": bottom_size,
                "images": images,
                "is_active": True,
                "store_name": "H&M",
                "name": product_name,
                "sex": "male",  # Since we're using men's category
                "tags": [],  # Empty array as default
                "description": description,
                "fit": fits[0] if fits else None,
                "measurements": measurements,
                "garment_length": garment_length,
                "waist_rise": waist_rise,
                "currency": currency,
                "rgb_color": rgb_color,
                "base_product_code": product_details.get("baseProductCode"),
                "assortment_type": product_details.get("assortmentTypeKey"),
                "supercategories": [cat.get("name") for cat in product_details.get("supercategories", [])]
            }
            
            print(f"Created item: {item['name']}")
            print(f"Category: {item['high_category']}/{item['specific_category']}")
            print(f"Color: {colors[0]} (RGB: {rgb_color})")
            print(f"Fit: {item['fit']}")
            print(f"Style: {style}")
            print(f"Images: {len(images)} total")
            
            # Count images by color
            color_counts = {}
            for img_url in images:
                if img_url in color_image_mapping:
                    color_name = color_image_mapping[img_url]["color_name"]
                    if color_name not in color_counts:
                        color_counts[color_name] = 0
                    color_counts[color_name] += 1
            
            for color_name, count in color_counts.items():
                print(f"  - {color_name}: {count} images")
                
            if measurements:
                print(f"Measurements: {', '.join(measurements)}")
            items.append(item)
            print(f"Added item {len(items)}/{num_items}")
        
        # Move to next page if we haven't found enough items
        if len(items) < num_items:
            page += 1
            time.sleep(1)  # Add delay to avoid rate limiting
        else:
            break
    
    if len(items) > 0:
        print(f"\nGenerated {len(items)} unique items successfully")
    else:
        print("\nCouldn't find any unique items after checking multiple pages")
    
    return items

def insert_test_items(items):
    """Insert test items into Supabase"""
    for item in items:
        try:
            response = supabase.table("hm_items").insert(item).execute()
            print(f"Inserted item {item['id']} successfully.")
        except Exception as e:
            print(f"Failed to insert item {item['id']}: {str(e)}")

if __name__ == "__main__":
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description='Generate H&M test data')
    parser.add_argument('num_items', type=int, help='Number of items to generate')
    args = parser.parse_args()
    
    print("Starting H&M test data generation...")
    print(f"Using API key: {HM_API_KEY[:5]}..." if HM_API_KEY else "No API key found!")
    
    # Use the command line argument
    test_items = generate_test_items(args.num_items)
    if test_items:
        print("\nInserting items into database...")
        insert_test_items(test_items)
        print(f"✅ {len(test_items)} items added successfully!")
    else:
        print("❌ No items were generated!") 