import numpy as np
from os import chdir

wd="/Users/moudiki/Documents/Python_Packages/teller"

chdir(wd)

import teller as tr
import pandas as pd

from sklearn import datasets
import numpy as np      
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


# import data
boston = datasets.load_boston()
X = np.delete(boston.data, 11, 1)
y = boston.target
col_names = np.append(np.delete(boston.feature_names, 11), 'MEDV')


# split  data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=123)
print(X_train.shape)
print(X_test.shape)


# fit a linear regression model 
regr = RandomForestRegressor(n_estimators=1000, random_state=123)
regr.fit(X_train, y_train)


# creating the explainer
df_test = pd.DataFrame(data = np.column_stack((X_test, y_test)), 
                       columns = col_names)
expr = tr.Explainer(obj=regr, df=df_test, target='MEDV')

# print(expr.get_params())

# expr = tr.Explainer(obj=regr, df=df_test, target='MEDV', n_jobs=2)

# fitting the explainer
expr.fit()


# heterogeneity of effects
print(expr.effects_)