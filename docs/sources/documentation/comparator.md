# Comparator

_Compare and explain predictions of two fitted models_

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/teller/explainer/comparator.py#L20)</span>

### Comparator


```python
teller.Comparator(obj1, obj2)
```


Class Comparator: Compare two models `obj1`, `obj2` ("estimators") based their predictions.
    
Attributes: 
   
    obj1: an object;
        fitted object containing methods `fit` and `predict`.

    obj2: an object;
       fitted object containing methods `fit` and `predict`.


----

<span style="float:right;">[[source]](https://github.com/Techtonique/teller/teller/explainer/comparator.py#L40)</span>

### summary


```python
Comparator.summary()
```


Summarise results of model comparison            

Args:  

    None  


----

