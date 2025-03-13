import os 
import math
import pandas as pd
import teller as tr
import numpy as np
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Load breast cancer dataset
breast_cancer = load_breast_cancer()
Z = breast_cancer.data
t = breast_cancer.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Train classifier
clf = ExtraTreesClassifier(
    n_estimators=250, 
    max_features=int(math.sqrt(X_train.shape[1])),
    random_state=24869
)
clf.fit(X_train, y_train)

# Create and fit the conformal explainer
# We'll explain predictions for class 1 (malignant)
expr = tr.ConformalExplainer(obj=clf, y_class=1)
expr.fit(X_test, X_names=breast_cancer.feature_names)

# Print summary of results
print(expr.summary()) 