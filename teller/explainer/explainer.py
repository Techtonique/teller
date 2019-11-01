import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from ..utils import deepcopy, is_factor, memoize, numerical_gradient


class Explainer(BaseEstimator):
    """class Explainer.
        
       Parameters
       ----------
       obj: object
           fitted object containing a method 'predict'
       df: data frame
           a data frame containing test set data + the target variable
       target: str
           name of the target variable (response, variable to be explained)
    """
    
    # construct the object -----

    def __init__(
        self,
        obj,
        df,
        target
    ):
        
        self.obj = obj 
        self.df = df
        self.target = target


    # fit the object -----
        
    def fit(self):        
        
        obj = self.obj 
        df = self.df
        target = self.target
        
        if isinstance(df, pd.DataFrame):                
            
            col_names = df.columns.values
            cond_training = (col_names != target)
            col_names = col_names[cond_training]
            
            X = df.iloc[:, cond_training].values
            
            # for classification, must be a prob
            y = df[target].values      
            
            y_hat = obj.predict(X)
            grad = numerical_gradient(obj.predict, X)
            
            res_df = pd.DataFrame(data=grad, 
                                  columns=col_names) 
    
            res_df_mean = res_df.mean()        
            res_df_std = res_df.std()        
            res_df_min = res_df.min()        
            res_df_max = res_df.max()        
            data = pd.concat([res_df_mean, res_df_std,
                                    res_df_min, res_df_max], 
                                    axis=1)
            
            df_effects = pd.DataFrame(data = data.values, 
                                      columns = ['mean', 'std', 'min', 'max'],
                                      index = col_names)
            
            # heterogeneity of effects
            self.y_mean = np.mean(y)
            sstot = np.sum((y - self.y_mean)**2)
            ssreg = np.sum((y_hat - self.y_mean)**2)
            ssres = np.sum((y - y_hat)**2)
            
            # heterogeneity of effects
            self.effects_ = df_effects.sort_values(by=['mean'])
            self.residuals_ = y - y_hat
            self.r_squared_ = 1 - ssres/sstot
            
            return self 
        
        raise ValueError("Dataset 'df' must be a data frame")
