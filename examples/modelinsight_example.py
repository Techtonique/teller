"""
Example usage of ModelInsight for regression and classification.

This script demonstrates how to use the ModelInsight class to generate interactive dashboards
and access model interpretation APIs for both regression and classification tasks.

Run this script after installing teller and its dependencies.

"""

from teller.modelinsight import ModelInsight
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

# Example 1: Regression
print("Creating regression example...")
X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = ModelInsight(
    model=model,
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    target_name="diabetes_progression"
)

# Generate and save dashboard
explainer.save_html("diabetes_dashboard.html", dark_mode=True)

# Example 2: Classification
print("\nCreating classification example...")
X, y = load_iris(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

explainer = ModelInsight(
    model=model,
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    target_name="iris_species"
)

explainer.save_html("iris_dashboard.html",
                   include_sections=['importance', 'metrics', 'predictions', 'confusion_matrix'],
                   dark_mode=False)

# Example 3: Using API methods
print("\n=== API Usage Examples ===")

# Get feature importance
importance = explainer.compute_feature_importance(method='permutation')
print("\nTop 5 Important Features:")
print(importance.nlargest(5, 'importance')[['feature', 'importance']])

# Get metrics
metrics = explainer.get_metrics()
print("\nModel Metrics:")
for key, value in metrics.items():
    print(f"  {key}: {value:.4f}")

# Explain single prediction
explanation = explainer.explain_prediction(0)
print("\nExplanation for first instance:")
print(f"  Prediction: {explanation['prediction']:.2f}")
print(f"  Actual: {explanation['actual']:.2f}")
print(f"  Features: {list(explanation['features'].keys())[:3]}...")

# Get feature statistics
stats = explainer.get_feature_statistics()
print("\nFeature Statistics (first feature):")
print(stats.iloc[0])

print("\n✅ Dashboards created successfully!")
print("📊 Open 'diabetes_dashboard.html' or 'iris_dashboard.html' in your browser")