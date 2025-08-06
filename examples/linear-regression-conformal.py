import os 
import teller as tr
from sklearn import datasets
from sklearn.linear_model import LinearRegression
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
regr = LinearRegression()
regr.fit(X_train, y_train)
print(f"columns: {col_names}")
print(f"coefficients: {regr.coef_}")

# creating the explainer
expr = tr.ConformalExplainer(obj=regr)

# heterogeneity of effects -----
# fitting the explainer
print("\n Conformal Explainer: -----------------")
expr.fit(X_test, X_names=col_names)
print(expr.summary())
expr.plot()

# Classical explainer 
print("\n Classical Explainer: -----------------")
expr = tr.Explainer(obj=regr)
expr.fit(X_test, y=y_test, X_names=col_names)
print(expr.summary())

