# Explainer

_Explain predictions for a fitted model_

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/explainer/explainer.py#L21)</span>

### Explainer


```python
teller.Explainer(obj, n_jobs=None, y_class=0, normalize=False)
```


Class Explainer: effects of features on the response.
    
Attributes:
   
    obj: an object;
        fitted object containing methods `fit` and `predict`

    n_jobs: an integer;
        number of jobs for parallel computing

    y_class: an integer;
        class whose probability has to be explained (for classification only)

    normalize: a boolean;
        whether the features must be normalized or not (changes the effects)


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/explainer/explainer.py#L59)</span>

### fit


```python
Explainer.fit(X, y, X_names, method="avg", type_ci="jackknife", scoring=None, level=95, col_inters=None)
```


Fit the explainer's attribute `obj` to training data (X, y).           

Args:

    X: array-like, shape = [n_samples, n_features]; 
        Training vectors, where n_samples is the number 
        of samples and n_features is the number of features.                

    y: array-like, shape = [n_samples, ]; Target values.

    X_names: {array-like}, shape = [n_features, ]; 
        Column names (strings) for training vectors.            

    method: str;
        Type of summary requested for effects. Either `avg` 
        (for average effects), `inters` (for interactions) 
        or `ci` (for effects including confidence intervals
        around them). 

    type_ci: str;
        Type of resampling for `method == 'ci'` (confidence 
        intervals around effects). Either `jackknife` 
        bootsrapping or `gaussian` (gaussian white noise with 
        standard deviation equal to `0.01` applied to the 
        features).

    scoring: str;
        measure of errors must be in ("explained_variance", 
        "neg_mean_absolute_error", "neg_mean_squared_error", 
        "neg_mean_squared_log_error", "neg_median_absolute_error", 
        "r2", "rmse") (default: "rmse").

    level: int; Level of confidence required for 
        `method == 'ci'` (in %).

    col_inters: str; Name of column for computing interactions.
               


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/blob/master/teller/explainer/explainer.py#L365)</span>

### summary


```python
Explainer.summary()
```


Summarise results 

    a method in class Explainer 

Args: 

    None   


----

