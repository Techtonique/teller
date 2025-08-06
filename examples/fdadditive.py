from sklearn.datasets import fetch_openml 
from sklearn.ensemble import RandomForestRegressor 
from teller import FDAdditiveExplainer
from sklearn.model_selection import train_test_split

X, y = fetch_openml("boston", as_frame=True, return_X_y=True)

X = X.select_dtypes(include=["number"]).drop(columns=["B"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Initialize explainer
explainer = FDAdditiveExplainer(rf.predict)

# Basic explanation (1st-order only)
basic_results = explainer.explain(X_test)

# Visualize
explainer.plot_attributions(basic_results["attributions"])
