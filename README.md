# H&M Clothing Recommender System

A hybrid recommendation system for H&M clothing items using LightFM. This system combines collaborative filtering with content-based approaches to provide personalized clothing recommendations.

## Features

- Hybrid recommendation approach (collaborative + content-based)
- Personalized recommendations based on user interactions
- Cold-start handling for new users
- Evaluation metrics for model performance
- Integration with Supabase database

## Prerequisites

- Python 3.7+
- Supabase account with tables for items and interactions
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Supabase credentials:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key
   ```

## Database Schema

The system expects two tables in your Supabase database:

1. `hm_items` - Contains information about clothing items
2. `hm_interactions` - Contains user interactions with items

## Usage

### Testing with a Specific User ID

```bash
python test_recommender.py <user_id>
```

Replace `<user_id>` with the ID of the user you want to generate recommendations for.

### Using the Recommender in Your Code

```python
# Import the recommender
from recommender import HMRecommender, init_supabase

# Initialize Supabase client
supabase = init_supabase()

# Initialize recommender
recommender = HMRecommender(supabase)

# Train model
recommender.train_model()

# Generate recommendations for a user
user_id = "e4d98795-62a8-4438-9136-5117ca6e6aee"
recommendations = recommender.recommend_for_user(user_id, n=10)

# Print recommendations
for i, rec in enumerate(recommendations, 1):
    item_details = rec.get('item_details', {})
    print(f"{i}. {item_details.get('name', 'Unknown')} - {item_details.get('brand', 'Unknown')}")
    print(f"   Score: {rec['score']:.4f}")
```

## Model Parameters

When training the model, you can adjust the following parameters:

- `num_components`: Number of latent factors (default: 30)
- `learning_rate`: Learning rate for the training algorithm (default: 0.05)
- `epochs`: Number of training epochs (default: 20)
- `loss`: Loss function ('warp', 'bpr', 'logistic', or 'warp-kos') (default: 'warp')

## Evaluation

The system includes an evaluation method that calculates the following metrics:

- Precision@10: Measures how many of the top 10 recommendations are relevant
- AUC: Area Under the ROC Curve, a measure of how well the model ranks relevant items

```python
metrics = recommender.evaluate_model()
print(metrics)
```

## Future Improvements

- Incremental learning for new items
- More sophisticated cold-start strategies
- Integration with image features for visual similarity
- A/B testing framework for recommendation strategies 