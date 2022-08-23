# PredictionInterval

_Obtain prediction intervals for input models_

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/predictioninterval/predictioninterval.py#L10)</span>

### PredictionInterval


```python
teller.PredictionInterval(obj, method="splitconformal", level=0.9, seed=123)
```


Class PredictionInterval: Obtain prediction intervals.
    
Attributes:
   
    obj: an object;
        fitted object containing methods `fit` and `predict`

    method: a string;
        method for constructing the prediction intervals. 
        Currently "splitconformal" (default) 

    level: a float;                
        Confidence level for prediction intervals. Default is 0.9, 
        equivalent to a miscoverage error of 0.1
    
    seed: an integer;
        Reproducibility of fit (there's a random split between fitting and calibration data)


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/predictioninterval/predictioninterval.py#L39)</span>

### fit


```python
PredictionInterval.fit(X, y)
```


Fit the PredictionInterval method to training data (X, y).           

Args:

    X: array-like, shape = [n_samples, n_features]; 
        Training set vectors, where n_samples is the number 
        of samples and n_features is the number of features.                

    y: array-like, shape = [n_samples, ]; Target values.
               


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/predictioninterval/predictioninterval.py#L64)</span>

### predict


```python
PredictionInterval.predict(X)
```


Obtain predictions and prediction intervals            

Args: 

    X: array-like, shape = [n_samples, n_features]; 
        Testing set vectors, where n_samples is the number 
        of samples and n_features is the number of features.                


----

