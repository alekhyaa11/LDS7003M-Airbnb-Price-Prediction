# Airbnb Price Prediction: London Market Analysis
**Module:** LDS7003M - Artificial Intelligence and Machine Learning (Component 2)

## 1. Project Overview
This project applies advanced machine learning techniques to predict the nightly price (`realSum`) of Airbnb listings in London. By analysing a dataset of over 4,000 listings, we aim to identify the key drivers of price (such as location, cleanliness, and capacity) and build a predictive model to assist hosts in dynamic pricing strategies.

## 2. Dataset Details
* **Source:** Airbnb London Weekday Prices Dataset.
* **Target Variable:** `realSum` (The total price of the listing).
* **Input Features:** 19 original features including `room_type`, `dist` (distance from city centre), `metro_dist`, and `guest_satisfaction_overall`.

## 3. Installation & Prerequisites
To reproduce this analysis, you will need **Python 3.x** and the following data science libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
How to Run the Analysis
Clone or Download this repository.

Ensure the dataset file Airbnb London Weekly Prices Dataset.csv is located in the same directory as main.py.

Execute the script via terminal or your preferred IDE: python main.py
Outputs:

Console logs showing data cleaning steps and cross-validation scores.

A generated plot prediction_plot.png visualizing Actual vs. Predicted prices.

Final performance metrics (RMSE, MAE, R²).

5. Methodology & Technical Approach
This analysis follows a standard Data Science pipeline:

A. Data Preprocessing
Cleaning: Rows with missing values were removed to ensure data integrity.

Encoding: Categorical variables (room_type, host_is_superhost) were transformed using Label Encoding.

Scaling: Feature scaling (StandardScaler) was applied to normalize the range of independent variables, crucial for algorithms like SVR.

B. Feature Engineering
We derived new features to capture hidden patterns:

Capacity_Per_Bedroom: Calculated as person_capacity / (bedrooms + 1). This differentiates between cramped and spacious listings.

Convenience_Score: An inverse composite of distance to the city centre and metro, prioritizing highly accessible locations.

C. Model Development
We implemented and compared three distinct algorithms:

Linear Regression: Established a baseline for performance.

Support Vector Regression (SVR): Tested for non-linear relationships in high-dimensional space.

Random Forest Regressor: Selected for its ability to handle complex interactions and resistance to overfitting.

D. Optimization
Recursive Feature Elimination (RFE): Used to identify the top 5 most significant features, reducing noise.

Hyperparameter Tuning: Applied GridSearchCV to optimize the Random Forest model (tuning n_estimators, max_depth, and min_samples_split).

6. Key Findings
Model Performance: The Random Forest Regressor outperformed Linear Regression and SVR, achieving the lowest RMSE.

Feature Importance: The analysis revealed that room_type, dist (Location), and person_capacity are the strongest predictors of price.

Ethical Note: All location data (lat/lng) was treated with caution. In a real-world deployment, precise coordinates would be obfuscated to protect host privacy.