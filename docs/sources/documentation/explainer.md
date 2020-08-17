<span style="float:right;">[[source]](https://github.com/Techtonique/teller/teller/explainer/explainer.py#L18)</span>

### Explainer


```python
teller.Explainer(obj, n_jobs=None, y_class=0, normalize=False)
```


**Class** Explainer: effects of features on the response.
 
__Arguments__

- __obj__: an object;
fitted object containing methods `fit` and `predict`
- __
n_jobs__: an integer;
number of jobs for parallel computing
- __
y_class__: an integer;
class whose probability has to be explained (for classification only)
- __
normalize__: a boolean;
whether the features must be normalized or not (changes the effects)

__Examples__

```python
print("hello world")
```                


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/teller/explainer/explainer.py#L62)</span>

### fit


```python
Explainer.fit(
    X, y, X_names, y_name, method="avg", type_ci="jackknife", scoring=None, level=95, col_inters=None
)
```


Fit the explainer's attribute `obj` to training data (X, y).           

__Arguments__

- __X__: array-like, shape = [n_samples, n_features]; 
 Training vectors, where n_samples is the number 
 of samples and n_features is the number of features.                
- __
 y__: array-like, shape = [n_samples, ]; Target values.
- __
 X_names__: {array-like}, shape = [n_features, ]; 
 Column names (strings) for training vectors.            
- __
 y_names__: str;
 Column name (string) for vector of target values. 
- __
 method__: str;
 Type of summary requested for effects. Either `avg` 
 (for average effects), `inters` (for interactions) 
 or `ci` (for effects including confidence intervals
 around them). 
- __
 type_ci__: str;
 Type of resampling for `method == 'ci'` (confidence 
 intervals around effects). Either `jackknife` 
 bootsrapping or `gaussian` (gaussian white noise with 
 standard deviation equal to `0.01` applied to the 
 features).
- __
 scoring__: str;
 measure of errors must be in ("explained_variance", 
 "neg_mean_absolute_error", "neg_mean_squared_error", 
 "neg_mean_squared_log_error", "neg_median_absolute_error", 
 "r2", "rmse") (default: "rmse").
- __
 level__: int; Level of confidence required for 
 `method == 'ci'` (in %).
- __
 col_inters__: str; Name of column for computing interactions.
        
# Examples 

```python
 print("hello world")
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/teller/explainer/explainer.py#L375)</span>

### summary


```python
Explainer.summary()
```


Summarise results of model comparison

a **method** in class Explainer 

__Examples __

```python
 print("hello world")
```


----

