# Modular H&M Recommender System

This is a modular implementation of the H&M recommender system, designed to provide personalized clothing recommendations based on user interactions and item features.

## Architecture

The system is organized into the following modules:

- **utils**: Utility functions for logging, initialization, etc.
- **data**: Data fetching and processing modules
- **models**: Recommendation model implementations
- **evaluation**: Metrics and evaluation utilities

## Key Components

1. **HMRecommender**: The main recommender class that ties everything together
2. **LightFMRecommender**: The underlying recommendation model based on LightFM
3. **Data Fetchers**: Modules for retrieving data from Supabase
4. **Data Processors**: Modules for processing and transforming data
5. **Evaluation Metrics**: Utilities for measuring recommender performance

## Usage

### Basic Usage

```python
from recommender import HMRecommender, init_supabase

# Initialize Supabase client
supabase = init_supabase()

# Initialize recommender
recommender = HMRecommender(supabase)

# Train model
recommender.train_model(
    num_components=30,
    learning_rate=0.05,
    epochs=20,
    loss='warp'
)

# Generate recommendations for a user
user_id = "e4d98795-62a8-4438-9136-5117ca6e6aee"
recommendations = recommender.recommend_for_user(user_id, n=10)

# Print recommendations
for i, rec in enumerate(recommendations, 1):
    item_details = rec.get('item_details', {})
    print(f"{i}. {item_details.get('name')} - Score: {rec['score']}")
```

### Command Line Usage

You can also use the provided command-line script:

```bash
python main.py --user_id e4d98795-62a8-4438-9136-5117ca6e6aee --num_recs 10
```

## Features

- **Personalized Recommendations**: Generates recommendations based on user preferences
- **Cold Start Handling**: Provides recommendations for new users with no interaction history
- **Dislike Filtering**: Avoids recommending items similar to those a user has disliked
- **Hybrid Approach**: Combines collaborative filtering with content-based features
- **Modular Design**: Easy to extend and maintain

## Requirements

- Python 3.6+
- LightFM
- Pandas
- NumPy
- Supabase Python Client

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables:
   - `SUPABASE_URL`: Your Supabase URL
   - `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key

## Database Schema

The system expects the following tables in Supabase:

1. **hm_items**: Contains item details
   - id: Item ID
   - name: Item name
   - brand: Brand name
   - price: Price
   - high_category: Main category (e.g., tops, bottoms)
   - specific_category: Specific category (e.g., shirts, jeans)
   - colors: Array of colors
   - styles: Array of styles
   - materials: Array of materials
   - sex: Gender (men, women, unisex)
   - condition: Item condition
   - season: Array of seasons
   - top_size, bottom_size, shoe_size: Size information
   - fit: Fit type
   - images: Array of image URLs

2. **hm_interactions**: Contains user-item interactions
   - user_id: User ID
   - item_id: Item ID
   - interaction_type: Type of interaction (like, dislike, superlike)
   - created_at: Timestamp of interaction
   - updated_at: Timestamp of last update 