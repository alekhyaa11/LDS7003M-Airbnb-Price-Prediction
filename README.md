# LDS7003M Group Project: Airbnb Price Prediction Analysis

## Project Overview
This project utilizes machine learning techniques to predict the nightly price (`realSum`) of Airbnb listings in London. By analysing features such as location, room type, cleanliness ratings, and guest satisfaction, we aim to build a robust regression model to estimate pricing.

## Dataset
**Source:** Airbnb London Weekday Prices Dataset
**Target Variable:** `realSum` (Total price)
**Key Features:** `room_type`, `dist` (distance to centre), `person_capacity`, `cleanliness_rating`.

## Methodology & Pipeline
1. **Data Preprocessing:**
   - **Cleaning:** Removed missing values and the unnecessary index column.
   - **Encoding:** Applied Label Encoding to categorical variables (`room_type`, `host_is_superhost`).
   - **Scaling:** Used `StandardScaler` to normalize numerical features for optimal model performance.

2. **Feature Engineering:**
   - Created `Capacity_Per_Bedroom`: To distinguish between crowded and spacious listings.
   - Created `Convenience_Score`: A composite metric combining distance to the city centre and metro stations.

3. **Model Development:**
   - Implemented three distinct algorithms:
     1. **Linear Regression:** As a baseline model.
     2. **Support Vector Regression (SVR):** For capturing non-linear relationships.
     3. **Random Forest Regressor:** For handling complex feature interactions.
   - **Validation:** Utilised 5-Fold Cross-Validation to ensure result consistency.

4. **Optimization:**
   - **Feature Selection:** Applied Recursive Feature Elimination (RFE) to identify the top 5 most significant predictors.
   - **Hyperparameter Tuning:** Used `GridSearchCV` to optimize the Random Forest model (tuning `n_estimators`, `max_depth`).

## How to Run the Analysis
**Prerequisites:** Python 3.x, pandas, numpy, scikit-learn, matplotlib, seaborn.

1. Ensure the dataset file `Airbnb London Weekly Prices Dataset.csv` is in the same directory as the script.
2. Run the main script:
   ```bash
   python main.py