import os
import random
import uuid
from supabase import create_client, Client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Data options matching DB constraints
brands = [
    "Nike", "Adidas", "Puma", "Under Armour", "Reebok",  # Sportswear brands
    "Zara", "H&M", "Uniqlo", "Mango", "Pull&Bear", "Bershka", "Massimo Dutti",  # Fast fashion
    "Levi's", "Wrangler", "Lee",  # Denim focused
    "The North Face", "Patagonia", "Columbia",  # Outdoor
    "Cartier", "Tiffany & Co", "Pandora",  # Luxury jewelry
    "Ray-Ban", "Oakley",  # Eyewear
    "Timberland", "Dr. Martens", "Converse", "Vans"  # Footwear
]

# Brand category restrictions
brand_categories = {
    "Nike": ["sportswear", "shoes", "accessories"],
    "Adidas": ["sportswear", "shoes", "accessories"],
    "Puma": ["sportswear", "shoes", "accessories"],
    "Under Armour": ["sportswear", "shoes", "accessories"],
    "Reebok": ["sportswear", "shoes"],
    "Zara": ["tops", "bottoms", "outerwear", "formal_wear"],
    "H&M": ["tops", "bottoms", "outerwear", "formal_wear", "accessories"],
    "Uniqlo": ["tops", "bottoms", "outerwear"],
    "Cartier": ["accessories"],
    "Tiffany & Co": ["accessories"],
    "Pandora": ["accessories"],
    "Ray-Ban": ["accessories"],
    "Oakley": ["accessories"],
    "Timberland": ["shoes"],
    "Dr. Martens": ["shoes"],
    "Converse": ["shoes"],
    "Vans": ["shoes"]
    # ... other brands will have access to all categories
}

categories = ["tops", "bottoms", "shoes", "outerwear", "accessories", "sportswear", "formal_wear"]
specific_categories = {
    "tops": ["t-shirts", "shirts", "tank_tops", "sweaters", "hoodies", "blouses", "polo_shirts"],
    "bottoms": ["pants", "jeans", "shorts", "skirts", "leggings"],
    "shoes": ["sneakers", "boots", "sandals", "flats", "heels", "loafers"],
    "outerwear": ["jackets", "coats", "blazers", "vests"],
    "accessories": ["scarves", "belts", "gloves", "sunglasses", "watches", "jewelry"],
    "sportswear": ["gym_tops", "gym_bottoms", "athletic_shoes", "bikinis", "one-piece_swimsuits"],
    "formal_wear": ["suits", "dresses", "tuxedos"]
}

colors = ["black", "white", "red", "blue", "green", "yellow", "purple", "pink", "orange", "brown", "grey", "multi"]
styles = ["casual", "formal", "sporty", "vintage", "streetwear", "bohemian", "minimalist", "preppy", "punk", "business", "chic"]
materials = [
    # Natural materials
    "cotton", "wool", "silk", "linen", "cashmere", "hemp",
    # Synthetic materials
    "polyester", "nylon", "spandex", "acrylic", "rayon",
    # Mixed materials
    "cotton_blend", "wool_blend",
    # Specific materials
    "denim", "leather", "suede", "canvas",
    # Luxury materials
    "sterling_silver", "gold_plated", "genuine_leather",
    # Technical materials
    "gore_tex", "dri_fit", "climacool"
]

# Material category restrictions
material_categories = {
    "sterling_silver": ["accessories"],
    "gold_plated": ["accessories"],
    "gore_tex": ["outerwear"],
    "dri_fit": ["sportswear"],
    "climacool": ["sportswear"]
}

# Define logical season combinations
season_combinations = [
    ["summer"],
    ["winter"],
    ["spring"],
    ["fall"],
    ["spring", "summer"],
    ["fall", "winter"],
    ["spring", "fall"],
    ["summer", "fall"],
    ["winter", "spring"]
]

# Generate test items
def generate_test_items(num_items=20):
    items = []
    for _ in range(num_items):
        # Generate high_category and specific_category with strict matching
        high_category = random.choice(list(specific_categories.keys()))
        specific_category = random.choice(specific_categories[high_category])

        # Select brand that matches the category
        valid_brands = [
            brand for brand in brands 
            if brand not in brand_categories or high_category in brand_categories[brand]
        ]
        brand = random.choice(valid_brands)

        # Select valid materials for the category
        valid_materials = [
            material for material in materials
            if material not in material_categories or high_category in material_categories[material]
        ]
        item_materials = [random.choice(valid_materials)]

        # Generate sizes based on high_category
        top_size = shoe_size = bottom_size = None
        if high_category == "tops":
            top_size = random.choice(["xs", "s", "m", "l", "xl", "xxl"])
        elif high_category == "shoes":
            shoe_size = str(random.randint(35, 45))  # EU shoe size
        elif high_category == "bottoms":
            bottom_size = f"W{random.randint(28, 36)}L{random.randint(30, 34)}"

        # Select a logical season combination
        season = random.choice(season_combinations)

        item = {
            "id": str(uuid.uuid4()),
            "brand": brand,
            "price": round(random.uniform(20, 200), 2),
            "high_category": high_category,
            "specific_category": specific_category,
            "colors": [random.choice(colors)],
            "styles": [random.choice(styles)],
            "materials": item_materials,
            "season": season,  # Now contains only logical season combinations
            "top_size": top_size,
            "shoe_size": shoe_size,
            "bottom_size": bottom_size,
            "images": ["https://via.placeholder.com/300"],
            "is_active": True,
            "store_name": "Test Store",
            "name": f"{specific_category.capitalize()}",
        }
        items.append(item)
    return items

# Insert test items into Supabase
def insert_test_items(items):
    for item in items:
        try:
            response = supabase.table("items").insert(item).execute()
            print(f"Inserted item {item['id']} successfully.")
        except Exception as e:
            print(f"Failed to insert item {item['id']}: {str(e)}")

# Generate and insert items
if __name__ == "__main__":
    test_items = generate_test_items(100)
    insert_test_items(test_items)
    print("✅ Test items added successfully!")
