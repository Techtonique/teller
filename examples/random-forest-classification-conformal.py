import os 
import math
import pandas as pd
import teller as tr
import numpy as np
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, make_classification

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

dataset = load_iris()
Z = dataset.data
t = dataset.target
np.random.seed(123)
X_train, X_test, y_train, y_test = train_test_split(Z, t, test_size=0.2)


print(X_train.shape)
print(X_test.shape)


clf1 = ExtraTreesClassifier(n_estimators=250, 
                              max_features=int(math.sqrt(X_train.shape[1])),
                              random_state=24869)


clf1.fit(X_train, y_train)

# creating the explainer
expr1 = tr.Explainer(obj=clf1, y_class=1, normalize=False)


# fitting the explainer (for heterogeneity of effects only)
expr1.fit(X_test, y_test, X_names=dataset.feature_names, 
          method="avg") 

# summary of results for the model
print(expr1.summary())


# creating the explainer
expr1 = tr.ConformalExplainer(obj=clf1, y_class=1)


# fitting the explainer (for heterogeneity of effects only)
expr1.fit(X_test, X_names=dataset.feature_names) 

# summary of results for the model
print(expr1.summary())

