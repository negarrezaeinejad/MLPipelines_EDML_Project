import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# --- 1. Data Generation ---
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

# --- 2. Pipeline Definition ---
def create_ml_pipeline():
    """
    Creates a scikit-learn machine learning pipeline for e-commerce data.
    """
    numerical_features = ['time_on_page', 'num_clicks']
    categorical_features = ['product_category']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', LabelEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('pass', 'passthrough', ['is_premium_member'])
        ],
        remainder='drop'
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    return pipeline

# --- 3. Execution ---

if __name__ == "__main__":
    # Data generation parameters
    num_total_samples = 15000
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    data_breakpoint_date = '2023-07-01'

    full_data = generate_ecommerce_data(num_total_samples, start_date, end_date, data_breakpoint_date)
    print(f"Total data generated: {len(full_data)} samples.")
    print("\nFirst 5 rows of generated data:")
    print(full_data.head())
    print("\nTarget variable distribution:")
    print(full_data['purchase'].value_counts(normalize=True))

    # Split data based on time
    train_data = full_data[full_data['timestamp'] < pd.to_datetime(data_breakpoint_date)].copy()
    test_data = full_data[full_data['timestamp'] >= pd.to_datetime(data_breakpoint_date)].copy()

    X_train = train_data.drop(['purchase', 'timestamp'], axis=1)
    y_train = train_data['purchase']

    X_test = test_data.drop(['purchase', 'timestamp'], axis=1)
    y_test = test_data['purchase']

    print(f"\nTraining data size: {len(X_train)} samples")
    print(f"Test data size: {len(X_test)} samples")

    pipeline = create_ml_pipeline()

    # Pre-encode categorical features for LabelEncoder compatibility before pipeline fit
    le_category = LabelEncoder()
    X_train['product_category'] = le_category.fit_transform(X_train['product_category'])
    trained_categories = set(le_category.classes_)

    pipeline.fit(X_train, y_train)
    print("\nModel training complete.")

    # Evaluate on training data
    y_train_pred = pipeline.predict(X_train)
    print("\n--- Evaluation on Training Data ---")
    print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print("Classification Report:\n", classification_report(y_train, y_train_pred, zero_division=0))

    # Prepare test data for prediction
    unseen_categories = set(X_test['product_category'].unique()) - trained_categories
    if unseen_categories:
        print(f"\nWarning: Test data contains categories not seen during training: {unseen_categories}. Assigning them to -1.")
        category_mapping = {category: le_category.transform([category])[0] for category in le_category.classes_}
        X_test['product_category'] = X_test['product_category'].apply(lambda x: category_mapping.get(x, -1))
    else:
        X_test['product_category'] = le_category.transform(X_test['product_category'])

    # Evaluate on test data
    y_test_pred = pipeline.predict(X_test)
    print("\n--- Evaluation on Test Data ---")
    print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print("Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))