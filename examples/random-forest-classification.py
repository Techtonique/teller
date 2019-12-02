import math
import pandas as pd
import teller as tr
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split


Default = pd.read_csv("/Users/moudiki/Documents/github_io_sandbox/Blog material/2019-11-29/Default.csv", 
                      sep=',')
X = Default.iloc[:,1:4].values
y = Default['defaultYes'].values
col_names = Default.columns.values
X_names = col_names[[1, 2, 3]]
y_name = col_names[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=80073)
print(X_train.shape)
print(X_test.shape)


clf1 = RandomForestClassifier(n_estimators=500, 
                            max_features=int(math.sqrt(X_train.shape[1])),
                            random_state=2989)


clf1.fit(X_train, y_train)


# creating the explainer
expr1 = tr.Explainer(obj=clf1, y_class=1, normalize=False)


# fitting the explainer (for heterogeneity of effects only)
expr1.fit(X_test, y_test, X_names=X_names, y_name=y_name, 
          method="avg") # put y_class and normalize in object creation


# confidence intervals and tests on marginal effects (Jackknife)
expr1.fit(X_test, y_test, X_names=X_names, y_name=y_name, 
          method="ci") # put y_class and normalize in object creation


# summary of results for the model
print(expr1.summary())


# see: https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
print(clf1.feature_importances_)

