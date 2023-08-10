import numpy as np      
import teller as tr
from sklearn import datasets, linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split


# import data
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
col_names = diabetes.feature_names


# split  data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=123)
print(X_train.shape)
print(X_test.shape)


# fit a linear regression model 
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
print(col_names)
print(regr.coef_)


# creating the explainer
expr = tr.Explainer(obj=regr, n_jobs=-1)


# print(expr.get_params())


# fitting the explainer
expr.fit(X_test, y_test, X_names=col_names)


# heterogeneity of effects
print(expr.summary())


expr.plot(what='average_effects')