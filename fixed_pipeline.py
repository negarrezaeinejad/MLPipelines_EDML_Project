import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

# --- 1. Data Generation (Same as original) ---
def generate_ecommerce_data(num_samples, start_date, end_date, breakpoint_date):
    """
    Generates synthetic e-commerce data.

    Parameters:
    - num_samples (int): Total number of samples to generate.
    - start_date (str): Start date for data generation (e.g., '2023-01-01').
    - end_date (str): End date for data generation (e.g., '2023-12-31').
    - breakpoint_date (str): A date marking a change in data characteristics.

    Returns:
    - pd.DataFrame: Generated dataset.
    """
    dates = pd.to_datetime(pd.date_range(start=start_date, end=end_date, periods=num_samples))
    df = pd.DataFrame({'timestamp': dates})

    np.random.seed(42) # for reproducibility

    # Base features
    df['time_on_page'] = np.random.normal(loc=60, scale=20, size=num_samples).clip(min=10, max=180)
    df['num_clicks'] = np.random.randint(low=1, high=30, size=num_samples)
    df['product_category'] = np.random.choice(['Electronics', 'Apparel', 'Books', 'HomeGoods'], size=num_samples)
    df['is_premium_member'] = np.random.choice([0, 1], size=num_samples, p=[0.7, 0.3])

    breakpoint_timestamp = pd.to_datetime(breakpoint_date)

    df['purchase'] = 0 # Initialize target

    # Define underlying relationships
    base_purchase_prob = 0.1
    param_A_influence_phase1 = 0.4
    param_B_influence_phase1 = 0.005

    param_A_influence_phase2 = 0.1
    param_B_influence_phase2 = 0.015

    for i, row in df.iterrows():
        prob = base_purchase_prob

        if row['timestamp'] < breakpoint_timestamp:
            # Phase 1 behavior
            if row['is_premium_member'] == 1:
                prob += param_A_influence_phase1
            prob += row['time_on_page'] * param_B_influence_phase1
            if row['num_clicks'] > 15:
                prob += 0.1
        else:
            # Phase 2 behavior
            if row['is_premium_member'] == 1:
                prob += param_A_influence_phase2
            prob += row['time_on_page'] * param_B_influence_phase2
            if row['num_clicks'] > 20:
                prob += 0.15

        prob = max(0, min(1, prob))
        df.loc[i, 'purchase'] = 1 if np.random.rand() < prob else 0

    return df

# --- 2. Fixed Pipeline Definition ---
def create_ml_pipeline():
    """
    Creates a properly configured scikit-learn machine learning pipeline for e-commerce data.
    """
    numerical_features = ['time_on_page', 'num_clicks']
    categorical_features = ['product_category']

    # Numerical preprocessing
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Categorical preprocessing - FIXED: Use OneHotEncoder instead of LabelEncoder
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('pass', 'passthrough', ['is_premium_member'])
        ],
        remainder='drop'
    )

    # Model configuration
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10,  # Prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2
    )

    # Create complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

# --- 3. Enhanced Evaluation Functions ---
def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    """
    Comprehensive model evaluation with multiple metrics.
    """
    print("=== MODEL EVALUATION ===")
    
    # Training performance
    y_train_pred = pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    
    # Test performance
    y_test_pred = pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\n--- Performance Summary ---")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Generalization Gap: {abs(train_accuracy - test_accuracy):.4f}")
    
    # Detailed test evaluation
    print(f"\n--- Detailed Test Results ---")
    print("Classification Report:")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))
    
    # Cross-validation on training data
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\n--- Cross-Validation Results ---")
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

def analyze_feature_importance(pipeline, feature_names):
    """
    Analyze and display feature importance from the trained model.
    """
    print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Get feature importance from the random forest
    rf_model = pipeline.named_steps['classifier']
    importance = rf_model.feature_importances_
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance Ranking:")
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return importance_df

# --- 4. Main Execution ---
if __name__ == "__main__":
    print("=== FIXED ML PIPELINE EXECUTION ===")
    
    # Data generation parameters
    num_total_samples = 15000
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    data_breakpoint_date = '2023-07-01'

    # Generate data
    print("\n1. Generating synthetic data...")
    full_data = generate_ecommerce_data(num_total_samples, start_date, end_date, data_breakpoint_date)
    print(f"Total data generated: {len(full_data)} samples.")
    
    # Data overview
    print("\nFirst 5 rows of generated data:")
    print(full_data.head())
    print("\nTarget variable distribution:")
    print(full_data['purchase'].value_counts(normalize=True))

    # Split data based on time (temporal split)
    print("\n2. Splitting data temporally...")
    train_data = full_data[full_data['timestamp'] < pd.to_datetime(data_breakpoint_date)].copy()
    test_data = full_data[full_data['timestamp'] >= pd.to_datetime(data_breakpoint_date)].copy()

    # Prepare features and targets - NO MANUAL PREPROCESSING
    X_train = train_data.drop(['purchase', 'timestamp'], axis=1)
    y_train = train_data['purchase']
    X_test = test_data.drop(['purchase', 'timestamp'], axis=1)
    y_test = test_data['purchase']

    print(f"Training data size: {len(X_train)} samples")
    print(f"Test data size: {len(X_test)} samples")
    print(f"Training target distribution: {y_train.value_counts(normalize=True).to_dict()}")
    print(f"Test target distribution: {y_test.value_counts(normalize=True).to_dict()}")

    # Create and train pipeline
    print("\n3. Training ML pipeline...")
    pipeline = create_ml_pipeline()
    
    # Fit the pipeline (handles all preprocessing automatically)
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # Evaluate model
    print("\n4. Evaluating model performance...")
    results = evaluate_model(pipeline, X_train, y_train, X_test, y_test)

    # Feature importance analysis
    print("\n5. Analyzing feature importance...")
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get feature names from the preprocessor
    numerical_features = ['time_on_page', 'num_clicks']
    categorical_features = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(['product_category'])
    passthrough_features = ['is_premium_member']
    
    all_feature_names = list(numerical_features) + list(categorical_features) + list(passthrough_features)
    
    importance_df = analyze_feature_importance(pipeline, all_feature_names)

    # Summary
    print("\n=== EXECUTION SUMMARY ===")
    print(f"✅ Pipeline executed successfully")
    print(f"✅ Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"✅ Cross-validation Score: {results['cv_mean']:.4f}")
    print(f"✅ Model shows {'good' if results['test_accuracy'] > 0.7 else 'moderate'} performance")
    print(f"✅ Generalization gap: {abs(results['train_accuracy'] - results['test_accuracy']):.4f}")
    
    if abs(results['train_accuracy'] - results['test_accuracy']) < 0.05:
        print("✅ Model generalizes well (low overfitting)")
    else:
        print("⚠️  Model may be overfitting (consider regularization)")