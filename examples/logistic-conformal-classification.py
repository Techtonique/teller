import os 
import math
import pandas as pd
import teller as tr
import numpy as np
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_iris, load_wine 

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

loaders = [load_breast_cancer, load_iris, load_wine]

names = ["breast_cancer", "iris", "wine"]

for name, dataset in zip(names, loaders): 
    print(f"data set: {name}")
    data = dataset()
    Z = data.data
    t = data.target
    np.random.seed(123)
    X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    # Train classifier
    clf = LogisticRegressionCV()
    clf.fit(X_train, y_train)
    # Create and fit the conformal explainer
    # We'll explain predictions for class 1 (malignant)
    expr = tr.ConformalExplainer(obj=clf, y_class=1)
    expr.fit(X_test, X_names=data.feature_names)
    # Print summary of results
    print(expr.summary()) 