import os 
import numpy as np      
import teller as tr
from sklearn import datasets, linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from teller.utils.numerical_gradient_conformal import finite_difference_interaction, finite_difference_sensitivity

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

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

print(finite_difference_sensitivity(regr, X_test, 
                                    n_jobs=1, show_progress=True))