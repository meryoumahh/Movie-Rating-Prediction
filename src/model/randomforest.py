from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
# --- STEP 4: MODELING ---
df_final = pd.read_csv(r'c:\Users\maata\Desktop\E-commerce\data\processed\clean_dataset_expAPI.csv')
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- STEP 4: ADVANCED MODELING ---

# 1. Setup Data
X = df_final.drop(['movieId', 'title', 'genres', 'avg_rating'], axis=1)
y = df_final['avg_rating']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("1. Data Split Complete.")
print(f"   Training on {X_train.shape[0]} movies.")
print(f"   Testing on {X_test.shape[0]} movies.")

# ---------------------------------------------------------
# PART A: GRID SEARCH WITH CROSS-VALIDATION (The Combo)
# ---------------------------------------------------------
# We define a 'grid' of settings we want to test.
# GridSearchCV will try EVERY combination using Cross-Validation (cv=3).
param_grid = {
    'n_estimators': [50, 100, 200],    
    'max_depth': [None, 10, 20],        
    'min_samples_split': [2, 10]        
}
rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=3, scoring='neg_root_mean_squared_error', 
                           n_jobs=-1, verbose=1)

print("\n2. Starting Grid Search (Tuning & CV)...")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("\n3. Best Parameters Found:")
print(grid_search.best_params_)

# ---------------------------------------------------------
# PART B: FINAL EVALUATION & PREDICTION COMPARISON
# ---------------------------------------------------------

# Predict using the BEST model
y_pred = best_model.predict(X_test)

# Calculate Errors
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("-" * 40)
print(f"FINAL TEST RMSE: {final_rmse:.4f}")
print(f"R2 Score: {r2:.4f} (Explains {r2*100:.1f}% of variance)")
print("-" * 40)

# ---------------------------------------------------------
# PART C: VISUALIZE ACTUAL VS PREDICTED (The Evidence)
# ---------------------------------------------------------

# Create a Comparison DataFrame
results_df = pd.DataFrame({
    'Actual Rating': y_test.values,
    'Predicted Rating': y_pred
})

# Calculate the error for each specific prediction
results_df['Error'] = results_df['Actual Rating'] - results_df['Predicted Rating']

print("\n--- Top 10 Prediction Examples ---")
print(results_df.head(10))

# OPTIONAL: Plot the Comparison
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual Rating', y='Predicted Rating', data=results_df, alpha=0.5)
plt.plot([0, 5], [0, 5], color='red', linestyle='--') # The "Perfect Prediction" line
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.show()

# Get the feature importance from the best model
importances = best_model.feature_importances_
feature_names = X.columns

# Create a DataFrame
feature_importance_df = pd.DataFrame({'Genre': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Genre', data=feature_importance_df, palette='viridis')
plt.title('Which Genres Impact Rating the Most?')
plt.xlabel('Importance Score')
plt.show()

# Export the trained model for use in other files
rf_model = best_model