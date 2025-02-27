"""
Main generator functions for H&M item generation.
"""

import uuid
import time

from .api import get_hm_data_with_cache, get_product_details_with_cache
from .mapping import map_hm_category, map_hm_color, map_hm_specific_category, map_style
from .utils import get_existing_items, is_duplicate_item, determine_seasons, validate_size, supabase

def generate_test_items(num_items=20, force_refresh=False):
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
    
    # Track how many items we've processed to avoid excessive API calls
    processed_items_count = 0
    max_items_to_process = num_items * 5  # Process at most 5x the number of items we need
    
    while len(items) < num_items and page <= max_pages and processed_items_count < max_items_to_process:
        print(f"\nFetching page {page} to find more unique products...")
        
        # Fetch products for the current page
        products_data = get_hm_data_with_cache("products/list", 
            f"country=us&lang=en&currentpage={page}&pagesize=100&categories=men_all",
            force_refresh=force_refresh if page == 1 else False  # Only force refresh the first page
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
        
        # Limit the number of products we process from this page
        remaining_needed = num_items - len(items)
        codes_to_process = new_product_codes[:min(remaining_needed * 3, len(new_product_codes))]
        print(f"Processing up to {len(codes_to_process)} products from this page")
        
        # Process the filtered products
        for product_code in codes_to_process:
            processed_items_count += 1
            
            if len(items) >= num_items:
                print(f"Reached target of {num_items} items, stopping processing")
                break
                
            if processed_items_count >= max_items_to_process:
                print(f"Processed maximum number of items ({max_items_to_process}), stopping to avoid excessive API calls")
                break
            
            # Get the basic product info from our mapping
            product = product_map[product_code]
            product_name = product["name"]
            
            print(f"Processing product {processed_items_count}: {product_name} (Code: {product_code})")
            
            # Check for basic duplicates that we can detect without API calls
            # For example, name/price combinations if price is available in the listing
            if "price" in product and product["price"].get("value") is not None:
                price_value = product["price"]["value"]
                name_price = (product_name, price_value)
                if name_price in existing_items['name_price_pairs'] or name_price in current_batch_name_price:
                    print(f"Skipping product with duplicate name and price (pre-check): {product_name}")
                    continue
                
            # Get detailed product information
            details = get_product_details_with_cache(product_code, force_refresh=force_refresh)
            
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
            
            # Get the primary color name and article code
            primary_color_name = product_details.get("color", {}).get("text", "")
            primary_article_code = product_details.get("code", "")
            
            # First check for direct galleryDetails in the product - these are the images for the primary article
            if "galleryDetails" in product_details:
                main_color = product_details.get("color", {}).get("text", "")
                main_color_lower = main_color.lower() if main_color else "unknown"
                main_rgb = product_details.get("color", {}).get("rgbColor", "")
                
                for gallery_item in product_details["galleryDetails"]:
                    if "baseUrl" in gallery_item:
                        image_url = gallery_item["baseUrl"]
                        images.append(image_url)
                        color_image_mapping[image_url] = {
                            "color_name": main_color,
                            "color_lower": main_color_lower,
                            "rgb_color": main_rgb
                        }
            
            # If we don't have any images from the primary article, look for the same article in articlesList
            if not images and "articlesList" in product_details:
                for article in product_details["articlesList"]:
                    # Only consider the same article code
                    if article.get("code") == primary_article_code:
                        variant_color = article.get("color", {}).get("text", "")
                        variant_color_lower = variant_color.lower() if variant_color else "unknown"
                        variant_rgb = article.get("color", {}).get("rgbColor", "")
                        
                        gallery_details = article.get("galleryDetails", [])
                        for gallery_item in gallery_details:
                            if "baseUrl" in gallery_item:
                                image_url = gallery_item["baseUrl"]
                                if image_url not in images:  # Avoid duplicates
                                    images.append(image_url)
                                    color_image_mapping[image_url] = {
                                        "color_name": variant_color,
                                        "color_lower": variant_color_lower,
                                        "rgb_color": variant_rgb
                                    }

            # Default image if none found
            if not images:
                print(f"No images found for {product_name}. No images will be uploaded.")
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
            
            # Use original color directly instead of mapping
            colors = []
            original_color_text = product_details.get("color", {}).get("text", "")
            if original_color_text:
                # Convert to lowercase for consistency
                colors = [original_color_text.lower()]
                print(f"Using original color: '{original_color_text.lower()}'")
            else:
                colors = ["unknown"]
                print("⚠️ No color information found - using 'unknown'")
                
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
            if colors[0] == "unknown":
                print(f"Color: {colors[0]}")
            else:
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