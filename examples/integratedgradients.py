import teller as tr
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np

# Load data
X, y = fetch_openml("boston", as_frame=True, return_X_y=True)
X = X.select_dtypes(include=["number"]).drop(columns=["B"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Initialize Integrated Gradients explainer
explainer = tr.IntegratedGradientsExplainer(rf.predict, baseline_method="mean", n_steps=50)

# Explain test set
results = explainer.explain(X_train.values, X_test.values, feature_names=list(X.columns))

# Print additivity diagnostics
print("Additivity RMSE:", results["additivity_rmse"])
print("Additivity max error:", results["additivity_max"])

# Optionally, inspect a few attributions and their sum
for i in range(3):
    print(f"\nSample {i}:")
    print("Prediction:", results["predictions"][i])
    print("Baseline + sum(attributions):", results["reconstructed"][i])
    print("Residual:", results["residuals"][i])
    print("Attributions:", results["attributions"][i])