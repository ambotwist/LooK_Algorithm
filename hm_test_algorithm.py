import os
from dotenv import load_dotenv
from supabase import create_client, Client
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import re
from generate_hm_test_data import generate_test_items

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -------------------- Fetch Data from Supabase -------------------- #
def fetch_products():
    try:
        response = supabase.table("hm_items").select("""
            id, brand, price, high_category, specific_category, 
            materials, colors, styles, tags, top_size, shoe_size, 
            bottom_size, sex, fit, measurements, garment_length, 
            waist_rise, rgb_color, base_product_code, assortment_type, 
            supercategories
        """).execute()
        if not response.data:
            print("No data returned from query")
            return pd.DataFrame()
        return pd.DataFrame(response.data)
    except Exception as e:
        print(f"Error fetching products: {e}")
        return pd.DataFrame()

def fetch_user_interactions():
    try:
        response = supabase.table("interactions").select("user_id, item_id, interaction_type").execute()
        if not response.data:
            print("No interaction data returned from query")
            return pd.DataFrame()
        return pd.DataFrame(response.data)
    except Exception as e:
        print(f"Error fetching user interactions: {e}")
        return pd.DataFrame()

def fetch_user_profile(user_id):
    try:
        response = supabase.table("user_profiles").select("*").eq("user_id", user_id).single().execute()
        return response.data if response.data else None
    except Exception as e:
        print(f"Error fetching user profile for user {user_id}: {e}")
        return None

# -------------------- Feature Engineering -------------------- #
def extract_rgb_features(rgb_color):
    """Convert hex color to RGB values"""
    if not rgb_color or not isinstance(rgb_color, str):
        return 0, 0, 0
    try:
        rgb = rgb_color.lstrip('#')
        r = int(rgb[:2], 16) / 255
        g = int(rgb[2:4], 16) / 255
        b = int(rgb[4:], 16) / 255
        return r, g, b
    except Exception as e:
        print(f"Error processing {rgb_color}: {e}")
        return 0, 0, 0

def extract_measurement_features(measurements):
    """Extract numerical values from measurement strings"""
    if not measurements or not isinstance(measurements, list):
        return 0
    try:
        numbers = []
        for meas in measurements:
            if isinstance(meas, str):
                found_nums = re.findall(r'\d+\.?\d*', meas)
                numbers.extend([float(num) for num in found_nums])
        return np.mean(numbers) if numbers else 0
    except Exception as e:
        print(f"Error processing measurements {measurements}: {e}")
        return 0

def engineer_features(products_df, for_training=True):
    """Consistent feature engineering pipeline for both training and prediction"""
    if products_df.empty:
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original dataframe
    df = products_df.copy()
    
    # Extract RGB features
    df[['r', 'g', 'b']] = pd.DataFrame(df['rgb_color'].apply(extract_rgb_features).tolist())
    
    # Extract measurement features
    df['measurement_value'] = df['measurements'].apply(extract_measurement_features)
    
    # Encode categorical features
    le = LabelEncoder()
    df['fit_encoded'] = le.fit_transform(df['fit'].fillna('None'))
    df['garment_length_encoded'] = le.fit_transform(df['garment_length'].fillna('None'))
    df['waist_rise_encoded'] = le.fit_transform(df['waist_rise'].fillna('None'))
    
    # Add price scaling
    scaler = MinMaxScaler()
    df['price_scaled'] = scaler.fit_transform(df[['price']])
    
    # Define base feature columns
    feature_columns = [
        'price_scaled', 'r', 'g', 'b', 'measurement_value',
        'fit_encoded', 'garment_length_encoded', 'waist_rise_encoded'
    ]
    
    # Add one-hot encoded features
    categorical_features = pd.get_dummies(df[[
        'brand', 'high_category', 'specific_category',
        'materials', 'colors', 'styles', 'tags',
        'assortment_type', 'supercategories'
    ]].explode(['materials', 'colors', 'styles', 'tags', 'supercategories']))
    
    # Combine all features
    features = pd.concat([
        df[feature_columns],
        categorical_features
    ], axis=1)
    
    # If this is for prediction, we need to return the original dataframe too
    if not for_training:
        return features, df
    
    return features

def prepare_ml_data(products, interactions):
    """Prepare data for machine learning by joining interactions with product features"""
    if interactions.empty or products.empty:
        print("Cannot prepare ML data: empty interactions or products dataframe")
        return None, None, None, None

    # Filter to only valid interaction types
    valid_interactions = interactions[interactions['interaction_type'].isin(['like', 'dislike'])]
    if valid_interactions.empty:
        print("Cannot prepare ML data: no valid like/dislike interactions found")
        return None, None, None, None

    # Convert interaction types to binary labels
    valid_interactions['liked'] = (valid_interactions['interaction_type'] == 'like').astype(int)
    
    # Merge interactions with product data
    try:
        data = valid_interactions.merge(products, left_on='item_id', right_on='id')
        if data.empty:
            print("Cannot prepare ML data: no matching products found for interactions")
            return None, None, None, None
    except Exception as e:
        print(f"Error merging interactions with products: {e}")
        return None, None, None, None
    
    # Extract labels before feature engineering
    labels = data['liked']
    
    # Use the consistent feature engineering pipeline
    features = engineer_features(data)
    
    # Split into training and testing sets
    return train_test_split(features, labels, test_size=0.3, random_state=42)

# -------------------- Train LightGBM Model -------------------- #
def train_model(X_train, y_train):
    """Train LightGBM model with cross-validation for more robust evaluation"""
    # Create dataset for LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    
    # Define model parameters
    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'metric': 'binary_logloss',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': 4
    }
    
    # Perform k-fold cross-validation
    print("Performing 5-fold cross-validation...")
    cv_results = lgb.cv(
        params,
        train_data,
        num_boost_round=100,
        nfold=5,
        stratified=True,
        early_stopping_rounds=10,
        verbose_eval=10,
        seed=42
    )
    
    # Print cross-validation results
    best_round = len(cv_results['binary_logloss-mean'])
    cv_score = cv_results['binary_logloss-mean'][-1]
    cv_std = cv_results['binary_logloss-stdv'][-1]
    print(f"Cross-validation score: {cv_score:.4f} ± {cv_std:.4f} at round {best_round}")
    
    # Train final model on all training data
    print("Training final model on all training data...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=best_round,
        valid_sets=[train_data],
        early_stopping_rounds=10
    )
    
    # Save model for future use
    model.save_model('lightgbm_model.txt')
    print("Model saved to 'lightgbm_model.txt'")
    
    return model

# -------------------- Recommend Products -------------------- #
def recommend_products(user_id, products, interactions, model, X_train):
    """Recommend products for a user based on their preferences and model predictions"""
    if products.empty:
        print(f"Cannot recommend products for user {user_id}: empty products dataframe")
        return pd.DataFrame()
        
    # Get user's previous interactions
    user_interactions = interactions[interactions['user_id'] == user_id]
    interacted_items = user_interactions['item_id'].tolist() if not user_interactions.empty else []
    
    # Filter out already interacted items
    filtered_products = products[~products['id'].isin(interacted_items)].copy()
    if filtered_products.empty:
        print(f"No uninteracted products available for user {user_id}")
        return pd.DataFrame()
    
    # Fetch user profile for personalization
    user_profile = fetch_user_profile(user_id)
    
    # Apply user-specific filters if profile exists and user is onboarded
    if user_profile and user_profile.get('is_onboarded'):
        # Filter by gender
        filtered_products = filtered_products[
            (filtered_products['sex'] == user_profile['gender']) | 
            (filtered_products['sex'] == 'unisex')
        ]
        
        # Filter by size
        if 'sizes' in user_profile:
            sizes = user_profile['sizes']
            size_mask = (
                ((filtered_products['high_category'] == 'tops') & filtered_products['top_size'].isin(sizes.get('tops', []))) |
                ((filtered_products['high_category'] == 'shoes') & (filtered_products['shoe_size'] == sizes.get('shoes'))) |
                ((filtered_products['high_category'] == 'bottoms') & (filtered_products['bottom_size'] == f"{sizes.get('bottoms', {}).get('waist')}{sizes.get('bottoms', {}).get('length')}")) |
                (~filtered_products['high_category'].isin(['tops', 'shoes', 'bottoms']))
            )
            filtered_products = filtered_products[size_mask]
        
        # Calculate style preference score
        if 'preferred_styles' in user_profile and user_profile['preferred_styles']:
            style_match = filtered_products['styles'].apply(
                lambda x: len(set(x) & set(user_profile['preferred_styles'])) if isinstance(x, list) else 0
            )
            filtered_products['style_score'] = style_match / max(len(user_profile['preferred_styles']), 1)
    
    if filtered_products.empty:
        print(f"No products match the user's {user_id} profile filters")
        return pd.DataFrame()
    
    # Use the consistent feature engineering pipeline
    prediction_features, filtered_products = engineer_features(filtered_products, for_training=False)
    
    # Ensure columns match training data
    for col in X_train.columns:
        if col not in prediction_features.columns:
            prediction_features[col] = 0
    prediction_features = prediction_features[X_train.columns]
    
    # Generate predictions
    try:
        filtered_products['recommendation_score'] = model.predict(prediction_features)
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return pd.DataFrame()
    
    # Calculate final score (combining model score with style preference if available)
    if 'style_score' in filtered_products.columns:
        filtered_products['final_score'] = filtered_products['recommendation_score'] * 0.7 + filtered_products['style_score'] * 0.3
    else:
        filtered_products['final_score'] = filtered_products['recommendation_score']
    
    # Sort and return recommendations
    recommendations = filtered_products.sort_values('final_score', ascending=False)
    return recommendations[[
        'id', 'brand', 'high_category', 'specific_category',
        'price', 'fit', 'garment_length', 'recommendation_score',
        'final_score'
    ]].head(10)

def select_important_features(feature_importance, threshold_percentage=1.0):
    """Select features that have importance above a certain threshold percentage"""
    # Calculate the threshold value based on percentage
    threshold = threshold_percentage / 100.0 * feature_importance['importance_percentage'].sum()
    
    # Select features above threshold
    important_features = feature_importance[
        feature_importance['importance_percentage'] >= threshold
    ]['feature'].tolist()
    
    print(f"Selected {len(important_features)} important features out of {len(feature_importance)} total features")
    print(f"Features with importance >= {threshold_percentage}% were kept")
    
    return important_features

# Update the analyze_feature_importance function to include feature selection
def analyze_feature_importance(model, X_train, threshold_percentage=1.0):
    """Analyze feature importance and select important features"""
    # Get feature importance
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importance()
    })
    
    # Calculate relative importance (percentage)
    total_importance = feature_importance['importance'].sum()
    feature_importance['importance_percentage'] = (feature_importance['importance'] / total_importance * 100)
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Group features by type
    feature_importance['feature_type'] = feature_importance['feature'].apply(lambda x: categorize_feature(x))
    
    # Calculate importance by feature type
    type_importance = feature_importance.groupby('feature_type')['importance'].sum().sort_values(ascending=False)
    type_importance_pct = (type_importance / total_importance * 100)
    
    # Save results
    feature_importance.to_csv('feature_importance_detailed.csv', index=False)
    
    # Select important features
    important_features = select_important_features(feature_importance, threshold_percentage)
    
    return feature_importance, type_importance_pct, important_features

def categorize_feature(feature_name):
    """Categorize features into types"""
    if feature_name in ['r', 'g', 'b']:
        return 'color_rgb'
    elif feature_name == 'measurement_value':
        return 'measurements'
    elif feature_name in ['price', 'price_scaled']:
        return 'price'
    elif feature_name.startswith('fit_'):
        return 'fit'
    elif feature_name.startswith('garment_length_'):
        return 'garment_length'
    elif feature_name.startswith('waist_rise_'):
        return 'waist_rise'
    elif any(feature_name.startswith(f"{cat}_") for cat in ['brand', 'high_category', 'specific_category']):
        return 'basic_categorization'
    elif any(feature_name.startswith(f"{cat}_") for cat in ['materials', 'colors', 'styles', 'tags']):
        return 'attributes'
    elif any(feature_name.startswith(f"{cat}_") for cat in ['assortment_type', 'supercategories']):
        return 'hm_categorization'
    else:
        return 'other'

# Add this function to get test items without inserting them
def get_test_batch(num_items=50):
    """Get a test batch of items without inserting them into database"""
    print(f"\nGenerating test batch of {num_items} items...")
    return generate_test_items(num_items)  # This makes API calls to H&M to get new items

def recommend_from_test_batch(user_id, test_batch, model, X_train):
    """Recommend items only from the test batch using the trained model"""
    if not test_batch:
        print(f"Cannot recommend from test batch for user {user_id}: empty test batch")
        return pd.DataFrame()
        
    # Get user's previous interactions
    user_interactions = fetch_user_interactions()
    user_interactions = user_interactions[user_interactions['user_id'] == user_id]
    
    # Prepare test batch features
    test_products = pd.DataFrame(test_batch)
    
    # Use the consistent feature engineering pipeline
    prediction_features, test_products = engineer_features(test_products, for_training=False)
    
    # Ensure columns match training data
    for col in X_train.columns:
        if col not in prediction_features.columns:
            prediction_features[col] = 0
    prediction_features = prediction_features[X_train.columns]
    
    # Generate predictions
    try:
        test_products['recommendation_score'] = model.predict(prediction_features)
    except Exception as e:
        print(f"Error generating predictions for test batch: {e}")
        return pd.DataFrame()
    
    # Get user profile for additional filtering and personalization
    user_profile = fetch_user_profile(user_id)
    if user_profile and user_profile.get('is_onboarded') and 'preferred_styles' in user_profile:
        # Apply user preferences
        preferred_styles = user_profile['preferred_styles']
        test_products['style_score'] = test_products['styles'].apply(
            lambda x: len(set(x) & set(preferred_styles)) / max(len(preferred_styles), 1) if isinstance(x, list) else 0
        )
        test_products['final_score'] = test_products['recommendation_score'] * 0.7 + test_products['style_score'] * 0.3
    else:
        test_products['final_score'] = test_products['recommendation_score']
    
    # Sort and return top 5 recommendations from test batch
    recommendations = test_products.sort_values('final_score', ascending=False)
    return recommendations[[
        'id', 'brand', 'high_category', 'specific_category',
        'price', 'fit', 'name', 'recommendation_score', 'final_score'
    ]].head(5)

def load_model(model_path='lightgbm_model.txt'):
    """Load a trained model from disk for production use"""
    try:
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found")
            return None
        model = lgb.Booster(model_file=model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def save_important_features(important_features, file_path='important_features.txt'):
    """Save the list of important features to a file for later use in production"""
    try:
        with open(file_path, 'w') as f:
            for feature in important_features:
                f.write(f"{feature}\n")
        print(f"Successfully saved {len(important_features)} important features to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving important features: {e}")
        return False

def load_important_features(file_path='important_features.txt'):
    """Load the list of important features from a file"""
    try:
        if not os.path.exists(file_path):
            print(f"Important features file {file_path} not found")
            return None
        with open(file_path, 'r') as f:
            important_features = [line.strip() for line in f.readlines()]
        print(f"Successfully loaded {len(important_features)} important features from {file_path}")
        return important_features
    except Exception as e:
        print(f"Error loading important features: {e}")
        return None

# -------------------- Main Execution -------------------- #
if __name__ == "__main__":
    print("Fetching data...")
    products_df = fetch_products()  # Get existing database items
    interactions_df = fetch_user_interactions()
    
    print("Getting test batch...")
    test_batch = get_test_batch(50)  # Get 50 test items
    
    # Display all items in test batch
    print("\nAll items in test batch:")
    test_batch_df = pd.DataFrame(test_batch)
    print(test_batch_df[[
        'name', 
        'high_category', 
        'specific_category',
        'price',
        'fit',
        'colors',
        'styles'
    ]].to_string())
    
    print("\nPreparing data for training...")
    result = prepare_ml_data(products_df, interactions_df)
    
    if result is None:
        print("Cannot train model: insufficient interaction data")
        
        # For production use, try to load existing model and important features
        print("\nAttempting to load existing model for production use...")
        model = load_model()
        important_features = load_important_features()
        
        if model is not None and important_features is not None:
            print("Successfully loaded model and important features for production use")
            
            # Test user recommendations with loaded model
            test_user_id = "e4d98795-62a8-4438-9136-5117ca6e6aee"  # Replace with your user ID
            print(f"\nGenerating recommendations from test batch for user {test_user_id}")
            
            # Create a dummy X_train with the important features for column alignment
            dummy_X_train = pd.DataFrame(columns=important_features)
            
            recommendations = recommend_from_test_batch(test_user_id, test_batch, model, dummy_X_train)
            print("\nTop 5 Recommendations from Test Batch:")
            print(recommendations[[
                'name',
                'high_category',
                'specific_category',
                'price',
                'fit',
                'recommendation_score',
                'final_score'
            ]].to_string())
    else:
        X_train, X_test, y_train, y_test = result
        
        if X_train is not None and y_train is not None and not y_train.empty:
            print("Training model...")
            model = train_model(X_train, y_train)
            
            # Evaluate model on test set
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred_binary)
            print(f"\nModel accuracy on test set: {accuracy:.4f}")
            
            # Enhanced feature importance analysis
            print("\nAnalyzing feature importance...")
            feature_importance, type_importance, important_features = analyze_feature_importance(model, X_train, threshold_percentage=1.0)
            
            # Save important features for later use
            save_important_features(important_features)
            
            print("\nTop 20 Most Important Individual Features:")
            print(feature_importance[['feature', 'importance_percentage']].head(20))
            
            print("\nFeature Importance by Category:")
            print(type_importance)
            
            # Filter X_train and X_test to only include important features
            X_train_important = X_train[important_features]
            X_test_important = X_test[important_features]
            
            # Train a more efficient model with only important features
            print("\nTraining a more efficient model with only important features...")
            efficient_model = train_model(X_train_important, y_train)
            
            # Evaluate efficient model
            y_pred_efficient = efficient_model.predict(X_test_important)
            y_pred_efficient_binary = (y_pred_efficient > 0.5).astype(int)
            efficient_accuracy = accuracy_score(y_test, y_pred_efficient_binary)
            print(f"\nEfficient model accuracy on test set: {efficient_accuracy:.4f}")
            
            # Compare model sizes
            original_model_size = os.path.getsize('lightgbm_model.txt') / 1024
            efficient_model_size = os.path.getsize('lightgbm_model.txt') / 1024  # Will overwrite the original
            print(f"\nOriginal model size: {original_model_size:.2f} KB")
            print(f"Efficient model size: {efficient_model_size:.2f} KB")
            print(f"Size reduction: {(1 - efficient_model_size/original_model_size) * 100:.2f}%")
            
            # Test user recommendations with efficient model
            test_user_id = "e4d98795-62a8-4438-9136-5117ca6e6aee"  # Replace with your user ID
            print(f"\nGenerating recommendations from test batch for user {test_user_id}")
            recommendations = recommend_from_test_batch(test_user_id, test_batch, efficient_model, X_train_important)
            print("\nTop 5 Recommendations from Test Batch:")
            print(recommendations[[
                'name',
                'high_category',
                'specific_category',
                'price',
                'fit',
                'recommendation_score',
                'final_score'
            ]].to_string())
            
            # Save feature importance visualizations
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                # Plot top 20 features
                plt.figure(figsize=(15, 8))
                sns.barplot(
                    data=feature_importance.head(20),
                    x='importance_percentage',
                    y='feature'
                )
                plt.title('Top 20 Most Important Features')
                plt.xlabel('Importance (%)')
                plt.tight_layout()
                plt.savefig('top_features_importance.png')
                
                # Plot feature importance by category
                plt.figure(figsize=(12, 6))
                type_importance.plot(kind='bar')
                plt.title('Feature Importance by Category')
                plt.xlabel('Feature Category')
                plt.ylabel('Importance (%)')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('feature_category_importance.png')
                
            except ImportError:
                print("Matplotlib and/or seaborn not installed. Skipping visualizations.")
        else:
            print("Cannot train model: training data is invalid or empty") 