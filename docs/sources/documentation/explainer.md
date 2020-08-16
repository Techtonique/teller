<span style="float:right;">[[source]](https://github.com/Techtonique/teller/teller/explainer/explainer.py#L18)</span>

### Explainer


```python
teller.Explainer(obj, n_jobs=None, y_class=0, normalize=False)
```


Class Explainer for: effects of features on the response.
 
Parameters
----------
obj: object
    fitted object containing methods `fit` and `predict`
n_jobs: int
    number of jobs for parallel computing
y_class: int
    class whose probability has to be explained (for classification only)
normalize:  boolean
    whether the features must be normalized or not (changes the effects)


----

