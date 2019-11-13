import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from ..utils import (
    deepcopy,
    get_code_pval,
    is_factor,
    memoize,
    numerical_gradient,
    numerical_gradient_jackknife,
    score_regression, 
    score_classification,
    t_test,
    var_test
)


class Comparator(BaseEstimator):
    """class Comparator.
        
       Parameters
       ----------
       obj1: object
           fitted object containing a method 'predict'
       obj2: object
           fitted object containing a method 'predict'
    """


    # construct the object -----
    def __init__(self, obj1, obj2):
        
        self.obj1 = obj1
        self.obj2 = obj2
    
    
    # summary of the object -----    
    def summary(self):
        
        assert (self.obj1.residuals_ is not None) & (self.obj2.residuals_ is not None), \
        "provided objects must be fitted first"

        assert self.obj1.scoring == self.obj2.scoring, \
        "scoring metrics must match for both objects"                
        
        
        # Gaussianity tests
        
        test_shapiro_obj1 = stats.shapiro(self.obj1.residuals_)
        test_shapiro_obj2 = stats.shapiro(self.obj2.residuals_)  
        
        test_jarque_bera_obj1 = stats.jarque_bera(self.obj1.residuals_)
        test_jarque_bera_obj2 = stats.jarque_bera(self.obj2.residuals_)                        
        
        print("\n")
        print(f"Scores ({self.obj1.scoring}): ")
        print(f"Object1: {np.round(self.obj1.score_, 3)}")
        print(f"Object2: {np.round(self.obj2.score_, 3)}")
        
        print("\n")
        print("R-squared: ")
        print("Object1: ")
        print(
            f"Multiple:  {np.round(self.obj1.r_squared_, 3)}, Adjusted:  {np.round(self.obj1.adj_r_squared_, 3)}"
        )
        print("Object2: ")
        print(
            f"Multiple:  {np.round(self.obj2.r_squared_, 3)}, Adjusted:  {np.round(self.obj2.adj_r_squared_, 3)}"
        )
        
        print("\n")
        print("Residuals: ")
        print("Object1: ")
        print(
            self.obj1.residuals_dist_.to_string(index=False)
        )
        print("Object2: ")
        print(
            self.obj2.residuals_dist_.to_string(index=False)
        )
        
        print("\n")
        print("Paired t-test (H0: mean(resids1) > mean(resids2) at 5%): ")
        t_test_obj = t_test(self.obj1.residuals_, self.obj2.residuals_)        
        print(f"statistic: {np.round(t_test_obj['statistic'], 5)}")
        print(f"p.value: {np.round(t_test_obj['p.value'], 5)}")
        print(f"conf. int: [{t_test_obj['f.int'][0]}, {np.round(t_test_obj['f.int'][1], 5)}]")
        print(f"mean of x: {np.round(t_test_obj['estimate']['mean of x'], 5)}")
        print(f"mean of y: {np.round(t_test_obj['estimate']['mean of y'], 5)}")
        print(f"alternative: {t_test_obj['alternative']}")
        
        
        if (self.obj1.ci_summary_ is not None):            
            
            df1_summary = self.obj1.ci_summary_[['Estimate', 'Std. Error', '']].sort_index(axis=0)
            df2_summary = self.obj2.ci_summary_[['Estimate', 'Std. Error', '']].sort_index(axis=0)
            
#            print("\n")                        
#            print(self.obj1.ci_summary_[['Estimate', '']])            
#            print(self.obj2.ci_summary_[['Estimate', '']])    
#            
#            print("\n")                        
#            print(df1_summary)            
#            print(df2_summary) 
            
            df_summary = pd.DataFrame(data=pd.concat([df1_summary, df2_summary], 
                                                     axis=1).values, 
                                      columns=['Estimate1', 'Std. Error1', 'Signif.', 'Estimate2', 'Std. Error2', 'Signif.'], 
                                      index=df1_summary.index)
            
            print("\n")
            print("Marginal effects: ")
            print(df_summary)                  
                        
        
#        print("\n")
#        print("Gaussianity tests on residuals (5%):")
#        
#        print("\n")
#        print("Shapiro-Wilk:")
#        print("Object 1: ")
#        print(f"Statistic: {test_shapiro_obj1[0]}, p-value: {test_shapiro_obj1[1]} {get_code_pval(test_shapiro_obj1[1])}")
#        print("Object 2: ")
#        print(f"Statistic: {test_shapiro_obj2[0]}, p-value: {test_shapiro_obj2[1]} {get_code_pval(test_shapiro_obj2[1])}")
#        
#        print("\n")
#        print("Jarque-Bera: ")
#        print("Object 1: ")
#        print(f"Statistic: {test_jarque_bera_obj1[0]}, p-value: {test_jarque_bera_obj1[1]} {get_code_pval(test_jarque_bera_obj1[1])}")
#        print("Object 2: ")
#        print(f"Statistic: {test_jarque_bera_obj2[0]}, p-value: {test_jarque_bera_obj2[1]} {get_code_pval(test_jarque_bera_obj2[1])}")
#                
#        
#        # Variance ratio tests
#        print("\n")
#        print("Variance ratio test on residuals (5%):")
#        print(var_test(self.obj1.residuals_, self.obj2.residuals_))
#        
#        
        print("\n")
        print(
            "Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘-’ 1"
        )
        
