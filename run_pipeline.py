import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle

# Load the dataset
df = pd.read_excel('Real estate valuation data set.xlsx')
df.head()

# Remove unnecessary columns
df = df.drop(["No", "X1 transaction date"], axis=1)

# Check for missing values and drop rows with NaN in any of the relevant columns
df.dropna(subset=['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores', 'Y house price of unit area'], inplace=True)

# Select features and the target variable
X_lat_long = df[['X5 latitude', 'X6 longitude']]
y = df['Y house price of unit area']

X_other_features = df[['X2 house age', 'X3 distance to the nearest MRT station', 'X4 number of convenience stores']]
 
# Check dimensions before split
print(f'Dimensioni di X_other_features: {X_other_features.shape}, Dimensioni di y: {y.shape}')

# Split into train and test sets for the first model
X_train_lat_long, X_test_lat_long, y_train, y_test = train_test_split(X_lat_long, y, test_size=0.2, random_state=42)

# First model
rf_lat_long = RandomForestRegressor(random_state=42)
param_grid_lat_long = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
grid_search_lat_long = GridSearchCV(rf_lat_long, param_grid_lat_long, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# Fit the first model
grid_search_lat_long.fit(X_train_lat_long, y_train)
best_model_lat_long = grid_search_lat_long.best_estimator_

# Save the first model using pickle
with open('best_model_lat_long.pickle', 'wb') as model_file:
    pickle.dump(best_model_lat_long, model_file)

print("Primo modello salvato con successo.")

# Check dimensions before split for the second model
print(f'Dimensioni di X_other_features: {X_other_features.shape}, Dimensioni di y: {y.shape}')

# Split into train and test sets for the second model
X_train_other_features, X_test_other_features, y_train_other, _ = train_test_split(X_other_features, y, test_size=0.2, random_state=42)

# Second model
rf_other_features = RandomForestRegressor(random_state=42)
param_grid_other_features = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
grid_search_other_features = GridSearchCV(rf_other_features, param_grid_other_features, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)

# Fit the second model
grid_search_other_features.fit(X_train_other_features, y_train_other)
best_model_other_features = grid_search_other_features.best_estimator_

# Save the second model using pickle
with open('best_model_other_features.pickle', 'wb') as model_file:
    pickle.dump(best_model_other_features, model_file)

print("Secondo modello salvato con successo.")