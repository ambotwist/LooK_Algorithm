import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")  # Make sure this is the service role key
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- Fetch Data from Supabase -------------------- #
def fetch_products():
    response = supabase.table("items").select("id, brand, price, high_category, specific_category, materials, colors, styles, tags").execute()
    return pd.DataFrame(response.data)

def fetch_user_interactions():
    response = supabase.table("interactions").select("user_id, item_id, interaction_type").execute()
    return pd.DataFrame(response.data)

def fetch_user_profiles():
    response = supabase.table("user_profiles").select("user_id, preferred_styles, sizes, gender").execute()
    return pd.DataFrame(response.data)

# -------------------- Prepare Data for Machine Learning -------------------- #
def prepare_ml_data(products, interactions):
    if len(interactions) == 0:
        return None, None, None, None

    valid_interactions = interactions[interactions['interaction_type'].isin(['like', 'dislike'])]
    
    if len(valid_interactions) == 0:
        return None, None, None, None

    valid_interactions['liked'] = (valid_interactions['interaction_type'] == 'like').astype(int)
    
    data = valid_interactions.merge(products, left_on='item_id', right_on='id')
    features = pd.get_dummies(data[['brand', 'high_category', 'specific_category', 'materials', 'colors', 'styles', 'tags']].explode(['materials', 'colors', 'styles', 'tags']))
    labels = data['liked']

    return train_test_split(features, labels, test_size=0.3, random_state=42)

# -------------------- Train ML Model -------------------- #
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# -------------------- Recommend Products for Specific User -------------------- #
def recommend_products(user_id, products, interactions, model):
    user_interactions = interactions[interactions['user_id'] == user_id]
    liked_item_ids = user_interactions[user_interactions['interaction_type'] == 'like']['item_id'].tolist()
    disliked_item_ids = user_interactions[user_interactions['interaction_type'] == 'dislike']['item_id'].tolist()

    filtered_products = products[~products['id'].isin(liked_item_ids + disliked_item_ids)]

    product_features = pd.get_dummies(filtered_products[['brand', 'high_category', 'specific_category', 'materials', 'colors', 'styles', 'tags']].explode(['materials', 'colors', 'styles', 'tags']))

    missing_cols = set(X_train.columns) - set(product_features.columns)
    for col in missing_cols:
        product_features[col] = 0
    product_features = product_features[X_train.columns]

    filtered_products['recommendation_score'] = model.predict_proba(product_features)[:, 1]
    recommendations = filtered_products.sort_values(by='recommendation_score', ascending=False)

    return recommendations[['id', 'brand', 'high_category', 'specific_category', 'price', 'recommendation_score']].head(10)

# -------------------- Main Execution -------------------- #
if __name__ == "__main__":
    products_df = fetch_products()
    interactions_df = fetch_user_interactions()
    user_profiles_df = fetch_user_profiles()

    result = prepare_ml_data(products_df, interactions_df)
    
    if result is None:
        print("Cannot train model: insufficient interaction data")
    else:
        X_train, X_test, y_train, y_test = result
        
        if X_train is not None and y_train is not None and len(y_train) > 0:
            model = train_model(X_train, y_train)
            y_pred = model.predict(X_test)
            print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

            user_id = "e4d98795-62a8-4438-9136-5117ca6e6aee"
            recommendations = recommend_products(user_id, products_df, interactions_df, model)
            print(f"Top 10 Recommendations for User {user_id}:")
            print(recommendations)
        else:
            print("Cannot train model: training data is invalid or empty")

    products_df.to_csv('supabase_products.csv', index=False)
    interactions_df.to_csv('supabase_interactions.csv', index=False)