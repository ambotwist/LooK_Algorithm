import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- Fetch Data from Supabase -------------------- #
def fetch_products():
    response = supabase.table("items").select("id, brand, price, high_category, specific_category, materials, colors, styles, tags, top_size, shoe_size, bottom_size, sex").execute()
    return pd.DataFrame(response.data)

def fetch_user_interactions():
    response = supabase.table("interactions").select("user_id, item_id, interaction_type").execute()
    return pd.DataFrame(response.data)

def fetch_user_profile(user_id):
    response = supabase.table("user_profiles").select("*").eq("user_id", user_id).single().execute()
    return response.data if response.data else None

# -------------------- Prepare Data for Machine Learning -------------------- #
def prepare_ml_data(products, interactions):
    if interactions.empty:
        return None, None, None, None

    valid_interactions = interactions[interactions['interaction_type'].isin(['like', 'dislike'])]
    if valid_interactions.empty:
        return None, None, None, None

    valid_interactions['liked'] = (valid_interactions['interaction_type'] == 'like').astype(int)
    data = valid_interactions.merge(products, left_on='item_id', right_on='id')

    # One-hot encode categorical features
    features = pd.get_dummies(data[['brand', 'high_category', 'specific_category', 'materials', 'colors', 'styles', 'tags']].explode(['materials', 'colors', 'styles', 'tags']))
    labels = data['liked']

    return train_test_split(features, labels, test_size=0.3, random_state=42)

# -------------------- Train LightGBM Model -------------------- #
def train_model(X_train, y_train):
    train_data = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'verbose': -1
    }
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

# -------------------- Recommend Products for Specific User -------------------- #
def recommend_products(user_id, products, interactions, model, X_train):
    user_interactions = interactions[interactions['user_id'] == user_id]
    liked_item_ids = user_interactions[user_interactions['interaction_type'] == 'like']['item_id'].tolist()
    disliked_item_ids = user_interactions[user_interactions['interaction_type'] == 'dislike']['item_id'].tolist()

    filtered_products = products[~products['id'].isin(liked_item_ids + disliked_item_ids)]

    # Fetch user profile
    user_profile = fetch_user_profile(user_id)

    if user_profile and user_profile['is_onboarded']:
        preferred_styles = user_profile['preferred_styles']
        gender = user_profile['gender']
        sizes = user_profile['sizes']

        # Filter by gender preference
        filtered_products = filtered_products[(filtered_products['sex'] == gender) | (filtered_products['sex'] == 'unisex')]

        # Boost scores for preferred styles
        def style_score(styles):
            return sum(1 for style in styles if style in preferred_styles)

        filtered_products['style_score'] = filtered_products['styles'].apply(style_score)

        # Size matching
        def size_match(row):
            if row['high_category'] == 'tops':
                return row['top_size'] in sizes['tops']
            elif row['high_category'] == 'shoes':
                return row['shoe_size'] == sizes['shoes']
            elif row['high_category'] == 'bottoms':
                return row['bottom_size'] == sizes['bottoms'].get('waist') + sizes['bottoms'].get('length')
            return True

        filtered_products = filtered_products[filtered_products.apply(size_match, axis=1)]

    # Prepare product features
    product_features = pd.get_dummies(filtered_products[['brand', 'high_category', 'specific_category', 'materials', 'colors', 'styles', 'tags']].explode(['materials', 'colors', 'styles', 'tags']))

    # Ensure matching columns between training and inference
    missing_cols = set(X_train.columns) - set(product_features.columns)
    for col in missing_cols:
        product_features[col] = 0
    product_features = product_features[X_train.columns]

    # Predict recommendation scores
    filtered_products['recommendation_score'] = model.predict(product_features)
    filtered_products['final_score'] = filtered_products['recommendation_score'] + filtered_products.get('style_score', 0)

    recommendations = filtered_products.sort_values(by='final_score', ascending=False)

    return recommendations[['id', 'brand', 'high_category', 'specific_category', 'price', 'recommendation_score', 'final_score']].head(10)

# -------------------- Main Execution -------------------- #
if __name__ == "__main__":
    products_df = fetch_products()
    interactions_df = fetch_user_interactions()

    result = prepare_ml_data(products_df, interactions_df)
    
    if result is None:
        print("Cannot train model: insufficient interaction data")
    else:
        X_train, X_test, y_train, y_test = result
        
        if X_train is not None and y_train is not None and not y_train.empty:
            model = train_model(X_train, y_train)
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

            user_id = "e4d98795-62a8-4438-9136-5117ca6e6aee"
            recommendations = recommend_products(user_id, products_df, interactions_df, model, X_train)
            print(f"Top 10 Recommendations for User {user_id}:")
            print(recommendations)
        else:
            print("Cannot train model: training data is invalid or empty")

    products_df.to_csv('supabase_products.csv', index=False)
    interactions_df.to_csv('supabase_interactions.csv', index=False)
