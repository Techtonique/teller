# PredictionInterval

_Obtain prediction intervals for input models_

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/predictioninterval/predictioninterval.py#L13)</span>

### PredictionInterval


```python
teller.PredictionInterval(obj, method="splitconformal", level=0.95, seed=123)
```


Class PredictionInterval: Obtain prediction intervals.
    
Attributes:
   
    obj: an object;
        fitted object containing methods `fit` and `predict`

    method: a string;
        method for constructing the prediction intervals. 
        Currently "splitconformal" (default) and "localconformal"

    level: a float;                
        Confidence level for prediction intervals. Default is 0.95, 
        equivalent to a miscoverage error of 0.05
    
    seed: an integer;
        Reproducibility of fit (there's a random split between fitting and calibration data)


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/predictioninterval/predictioninterval.py#L43)</span>

### fit


```python
PredictionInterval.fit(X, y)
```


Fit the `method` to training data (X, y).           

Args:

    X: array-like, shape = [n_samples, n_features]; 
        Training set vectors, where n_samples is the number 
        of samples and n_features is the number of features.                

    y: array-like, shape = [n_samples, ]; Target values.
               


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/predictioninterval/predictioninterval.py#L89)</span>

### predict


```python
PredictionInterval.predict(X, return_pi=False)
```


Obtain predictions and prediction intervals            

Args: 

    X: array-like, shape = [n_samples, n_features]; 
        Testing set vectors, where n_samples is the number 
        of samples and n_features is the number of features. 

    return_pi: boolean               
        Whether the prediction interval is returned or not. 
        Default is False, for compatibility with other _estimators_.
        If True, a tuple containing the predictions + lower and upper 
        bounds is returned.


----

