import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, r2_score

# =====================================================================
# 1. LOAD THE CSV DATASET
# =====================================================================
try:
    df = pd.read_csv('serene_mock_data.csv')
    print(" Successfully loaded serene_mock_data.csv")
except FileNotFoundError:
    print(" Error: serene_mock_data.csv not found in this folder!")
    exit()

# Separate what we want to predict (stress_score) from the input features
X_raw = df.copy()
y = X_raw.pop('stress_score')  # This is our target variable
X_raw.pop('user_id')           # Drop user_id since it's just metadata

# =====================================================================
# 2. CONVERT TEXT STRESSORS TO NUMBERS
# =====================================================================
# This turns the comma-separated string back into a clean Python list of words
X_raw['stressor_list'] = X_raw['primary_stressors'].apply(lambda x: [s.strip() for s in str(x).split(',')])

# MultiLabelBinarizer creates a new column for EVERY unique stressor tag.
# If a user has 'Burnout', their 'Burnout' column gets a 1, otherwise a 0.
mlb = MultiLabelBinarizer()
stressor_encoded = mlb.fit_transform(X_raw['stressor_list'])
stressor_df = pd.DataFrame(stressor_encoded, columns=mlb.classes_)

# =====================================================================
# 3. CONVERT PERSONAS & GENDERS TO NUMBERS (One-Hot Encoding)
# =====================================================================
# This creates separate 1 and 0 columns for things like persona_Teen, persona_Corporate, etc.
categorical_df = pd.get_dummies(X_raw[['persona', 'gender']], drop_first=False, dtype=int)

# Combine numerical features, categorical features, and our text features into one final table
numerical_features = X_raw[['age', 'sleep_avg_hours', 'screen_time_hours', 'work_study_hours', 'wellness_points']]
X = pd.concat([numerical_features, categorical_df, stressor_df], axis=1)

# Save the exact order of columns so our backend API knows the layout later
feature_columns = X.columns.tolist()

# =====================================================================
# 4. SPLIT DATA & TRAIN THE XGBOOST MODEL
# =====================================================================
# Split data: 80% to train the model, 20% to test if it actually learned correctly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training XGBoost on {X_train.shape[0]} samples with {X_train.shape[1]} features...")

# Initialize the XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.08,
    random_state=42
)

# Train the model!
model.fit(X_train, y_train)

# =====================================================================
# 5. EVALUATE PERFORMANCE
# =====================================================================
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)

print("\n--- Training Evaluation Results ---")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, predictions):.4f}")
print(f"R-squared (R2) Score: {r2:.4f}")
print(f" The model successfully learned {r2*100:.1f}% of the stress patterns!")

# =====================================================================
# 6. SAVE EVERYTHING TO DISK (Artifacts)
# =====================================================================
print("\nSaving your trained assets to your folder...")

# Save the brain of the ML model
model.save_model('xgboost_stress_model.json')

# Save the text encoder so we can convert live chat data the exact same way later
with open('stressor_encoder.pkl', 'wb') as f:
    pickle.dump(mlb, f)

# Save the layout order of the columns
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("✓ All files saved. Your model is fully trained!")
