# ==========================================
# LDS7003M: AI and Machine Learning Group Project
# Component 2: Airbnb Price Prediction (Regression)
# ==========================================

# 1. IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing & Selection
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import RFE

# Models (Requirement: At least 3 algorithms)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 2. DATA EXPLORATION AND PREPROCESSING
# ==========================================

# Load Data
# NOTE: Ensure the file is named exactly as below
csv_filename = 'Airbnb London Weekly Prices Dataset.csv'
try:
    df = pd.read_csv(csv_filename)
    print("Data Loaded Successfully.")
except FileNotFoundError:
    print(f"Error: '{csv_filename}' not found. Please download from Moodle.")
    exit()

# 2.1 Data Cleaning
# Remove index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop('Unnamed: 0', axis=1)

# Handle Missing Data
# Strategy: Drop rows with missing values (dataset is large enough)
print(f"Original shape: {df.shape}")
df = df.dropna()
print(f"Shape after dropping missing values: {df.shape}")

# 2.2 Feature Engineering (Requirement: Create new relevant features)
# Feature 1: 'Capacity_Per_Bedroom' - Helps identify crowded listings vs spacious ones
# Adding 1 to bedrooms to avoid division by zero for studio apartments
df['Capacity_Per_Bedroom'] = df['person_capacity'] / (df['bedrooms'] + 1)

# Feature 2: 'Convenience_Score' - Inverse of distance to center and metro
# Higher score means better location
df['Convenience_Score'] = 1 / (df['dist'] + df['metro_dist'] + 0.1)

# 2.3 Encoding Categorical Variables
label_encoder = LabelEncoder()

# Encode 'room_type'
df['room_type'] = label_encoder.fit_transform(df['room_type'])

# Encode Boolean/Object columns
for col in ['host_is_superhost', 'room_shared', 'room_private']:
    if df[col].dtype == 'object' or df[col].dtype == 'bool':
        df[col] = label_encoder.fit_transform(df[col].astype(str))

# 2.4 Exploratory Data Analysis (EDA) - Correlation
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
# plt.show() # Uncomment to view plot

# ==========================================
# 3. SPLIT AND SCALE
# ==========================================

# Define Target (y) and Features (X)
# Target is 'realSum' (the price of the listing)
target_col = 'realSum'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Scaling and Normalization (Requirement: Scale data)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ==========================================
# 4. MODEL DEVELOPMENT
# ==========================================

# Initialize 3 Different Algorithms
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regression (SVR)": SVR(kernel='rbf'),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=50, random_state=42)
}

# K-Fold Cross Validation (Requirement: Assess performance)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
print("\n--- K-Fold Cross-Validation Results (RMSE) ---")
for name, model in models.items():
    # cv_results returns negative MSE, so we negate it and take sqrt
    cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    results[name] = rmse_scores.mean()
    print(f"{name}: {results[name]:.2f}")

# ==========================================
# 5. MODEL EVALUATION & OPTIMIZATION
# ==========================================

# 5.1 Hyperparameter Tuning (GridSearch)
# Tuning Random Forest (likely the best performer)
print("\n--- Tuning Hyperparameters for Random Forest ---")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Using a smaller CV for GridSearch to save time
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, 
                           scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# 5.2 Feature Selection (RFE)
# Using Recursive Feature Elimination to find top 5 features
print("\n--- Feature Selection (RFE) ---")
rfe = RFE(estimator=LinearRegression(), n_features_to_select=5)
rfe.fit(X_train, y_train)

# Identify selected features
selected_features_mask = rfe.support_
selected_features = X.columns[selected_features_mask]
print("Top 5 Features selected by RFE:")
print(list(selected_features))

# ==========================================
# 6. FINAL EVALUATION
# ==========================================

# Predict on Test Set using the Optimised Model
y_pred = best_rf_model.predict(X_test)

# Calculate Regression Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Final Model Performance (Tuned Random Forest) ---")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"MAE (Mean Absolute Error):      {mae:.2f}")
print(f"R-Squared (Accuracy):           {r2:.2f}")

# Visualisation: Actual vs Predicted
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Airbnb Prices")
plt.savefig('prediction_plot.png')
print("Plot saved as 'prediction_plot.png'")