![teller logo](the-teller.png)

<hr>  

There is an increasing need for __transparency__ and __fairness__ in Machine Learning (ML) models  predictions. Consider for example a banker who has to explain to a client why his/her loan application is rejected, or a health professional who must explain what constitutes his/her diagnostic. Some ML models are indeed very accurate, but are considered  hard to explain, relatively to popular linear models. We do not want to sacrifice this high accuracy to explainability.  Hence: __ML explainability__. There are a lot of ML explainability tools out there, _in the wild_ (don't take my word for it).

The `teller` is a __model-agnostic tool for ML explainability__ - agnostic, as long as  this model possesses methods `fit` and `predict`. The `teller`'s philosophy is to rely on [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) to explain ML models predictions: a little increase in model's explanatory variables + a little decrease, and we can obtain approximate sensitivities of its predictions to changes in these explanatory variables. 

## Installation 

- Currently from Github, for the development version: 

```bash
pip install git+https://github.com/thierrymoudiki/teller.git
```

## Package description

This notebook will be a good introduction:

[thierrymoudiki_011119_boston_housing.ipynb](/teller/demo/thierrymoudiki_011119_boston_housing.ipynb)

Two models are used in the notebook: a __linear model__ and a [Random Forest](https://en.wikipedia.org/wiki/Random_forest) (here, the _black-box_ model). The most straightforward way to illustrate the `teller` is to use a linear model. In this case, the effects of model covariates on the response can be directly related to the linear model's coefficients.


## Contributing

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first. In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting: 

```bash
pip install black
black --line-length=60 file_submitted_for_pr.py
```

## Dependencies 

- Numpy
- Pandas
- Scipy
- scikit-learn


## References

- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (1992). Numerical recipes in C (Vol. 2). Cambridge: Cambridge university press.
- Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 2019-01-04]
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 

