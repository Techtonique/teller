import teller as tr
from sklearn import datasets
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
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
regr = RandomForestRegressor(n_estimators=1000, random_state=123)
regr.fit(X_train, y_train)

# creating the explainer
expr = tr.Explainer(obj=regr, n_jobs = -1)

# heterogeneity of effects -----
# fitting the explainer
expr.fit(X_test, y_test, X_names=col_names, 
          method="avg", type_ci="gaussian")
print(expr.summary())


# confidence int. and tests on effects -----
expr.fit(X_test, y_test, X_names=col_names, 
         method="ci", type_ci="gaussian")
print(expr.summary())


# BROKEN for now
# # interactions -----

# varx = "bmi"
# expr.fit(X_test, y_test, X_names=col_names,         
#          col_inters = varx, method="inters")
# print(expr.summary())


# varx = "age"
# expr.fit(X_test, y_test, X_names=col_names, 
#          col_inters = varx, method="inters")
# print(expr.summary())
