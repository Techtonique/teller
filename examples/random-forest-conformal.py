import os 
import teller as tr
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# import data
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
col_names = diabetes.feature_names

# split  data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=123)

# fit a linear regression model 
regr = RandomForestRegressor(n_estimators=250, random_state=123)
regr.fit(X_train, y_train)

# creating the explainer
expr = tr.ConformalExplainer(obj=regr)

# heterogeneity of effects -----
# fitting the explainer
expr.fit(X_test, X_names=col_names)
print(expr.summary())

