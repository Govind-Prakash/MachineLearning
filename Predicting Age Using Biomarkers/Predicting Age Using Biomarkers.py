#Step 1: Simulated Dataset Creation For simplicity, let’s create a synthetic dataset representing biomarker levels and corresponding ages. This will act as our placeholder until you have real-world data.Code to Simulate Dataset:

import numpy as np
import pandas as pd

# Simulate data
np.random.seed(42)  # Ensure reproducibility
n_samples = 500  # Number of samples
n_features = 10  # Number of biomarkers

# Generate random biomarker levels (values between 0 and 1)
biomarkers = np.random.rand(n_samples, n_features)

# Create age as a combination of biomarker levels with some noise
true_weights = np.random.uniform(1, 5, size=n_features)  # True weights for biomarkers
ages = biomarkers @ true_weights + np.random.normal(0, 2, size=n_samples)  # Linear relationship + noise

# Convert to DataFrame
columns = [f"biomarker_{i+1}" for i in range(n_features)]
data = pd.DataFrame(biomarkers, columns=columns)
data['age'] = ages

print(data.head())  # Print the first few rows of the dataset


# Display the dataset
#import ace_tools as tools; tools.display_dataframe_to_user("Synthetic Biomarker Dataset", data)
# Display the dataset
print(data.head())  # Option 1: Print the first few rows

# OR save it to a file
data.to_csv("synthetic_biomarker_dataset.csv", index=False)  # Option 2: Save to a CSV
print("Dataset saved as synthetic_biomarker_dataset.csv")

# OR if in Jupyter Notebook
# from IPython.display import display
# display(data)

#Step 2: Preprocessing

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = data.iloc[:, :-1]  # All biomarkers
y = data['age']  # Age

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Baseline modelling

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R² Score: {r2:.2f}")


# Advanced Modeling
from sklearn.ensemble import RandomForestRegressor

# Train the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest MSE: {mse_rf:.2f}")
print(f"Random Forest R² Score: {r2_rf:.2f}")

# Feature importance

import matplotlib.pyplot as plt

# Get feature importances
feature_importances = rf_model.feature_importances_
sorted_indices = np.argsort(feature_importances)[::-1]

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
plt.xticks(range(len(feature_importances)), [columns[i] for i in sorted_indices], rotation=45)
plt.title("Feature Importance in Predicting Age")
plt.show()
