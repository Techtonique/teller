from locale import normalize
import numpy as np
import pickle
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from ..nonconformist import IcpRegressor
from ..nonconformist import RegressorNc 
from ..nonconformist import RegressorNormalizer, AbsErrorErrFunc
from ..utils import Progbar


class PredictionInterval(BaseEstimator, RegressorMixin):
    """Class PredictionInterval: Obtain prediction intervals.
        
    Attributes:
       
        obj: an object;
            fitted object containing methods `fit` and `predict`

        method: a string;
            method for constructing the prediction intervals. 
            Currently "splitconformal" (default) and "localconformal"

        level: a float;                
            Confidence level for prediction intervals. Default is 0.9, 
            equivalent to a miscoverage error of 0.1
        
        seed: an integer;
            Reproducibility of fit (there's a random split between fitting and calibration data)
    """

    def __init__(self, obj, method="splitconformal", level=0.9, seed=123):

        self.obj = obj
        self.method = method
        self.level = level
        self.seed = seed
        self.icp_ = None


    def fit(self, X, y):
        """Fit the `method` to training data (X, y).           
        
        Args:

            X: array-like, shape = [n_samples, n_features]; 
                Training set vectors, where n_samples is the number 
                of samples and n_features is the number of features.                

            y: array-like, shape = [n_samples, ]; Target values.
                       
        """       

        X_train, X_calibration, y_train, y_calibration = train_test_split(X, y, 
                                                    test_size=0.5, random_state=self.seed)

        if self.method == "splitconformal":             
            nc = RegressorNc(self.obj, AbsErrorErrFunc())            

        if self.method == "localconformal":
            mad_estimator = pickle.loads(pickle.dumps(self.obj, -1))
            normalizer = RegressorNormalizer(self.obj, mad_estimator, AbsErrorErrFunc())
            nc = RegressorNc(self.obj, AbsErrorErrFunc(), normalizer)
        
        self.icp_ = IcpRegressor(nc)
        self.icp_.fit(X_train, y_train) 
        self.icp_.calibrate(X_calibration, y_calibration)

        return self


    def predict(self, X, return_pi=False):
        """Obtain predictions and prediction intervals            

        Args: 

            X: array-like, shape = [n_samples, n_features]; 
                Testing set vectors, where n_samples is the number 
                of samples and n_features is the number of features. 

            return_pi: boolean               
                Whether the prediction interval is returned or not. 
                Default is False, for compatibility with other _estimators_.
                If True, a tuple containing the predictions + lower and upper 
                bounds is returned.

        """                

        pred = self.obj.predict(X)
        
        if return_pi:
            predictions = self.icp_.predict(X, significance = 1-self.level)
            return pred, predictions[:, 0], predictions[:, 1]
        else:
            return pred

