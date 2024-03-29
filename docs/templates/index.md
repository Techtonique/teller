# teller | <a class="github-button" href="https://github.com/Techtonique/teller/stargazers" data-color-scheme="no-preference: light; light: light; dark: dark;" data-size="large" aria-label="Star the teller /the teller  on GitHub">Star</a>

![PyPI](https://img.shields.io/pypi/v/the-teller) [![PyPI - License](https://img.shields.io/pypi/l/the-teller)](https://github.com/thierrymoudiki/teller/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/the-teller)](https://pepy.tech/project/the-teller) [![Last Commit](https://img.shields.io/github/last-commit/Techtonique/teller)](https://github.com/Techtonique/teller)


Welcome to the __teller__'s website.

There is an increasing need for __transparency__ and __fairness__ in Machine Learning (ML) models  predictions. Consider for example a banker who has to explain to a client why his/her loan application is rejected, or a healthcare professional who must explain what constitutes his/her diagnosis. Some ML models are indeed very accurate, but are considered to be hard to explain, relatively to popular linear models. 


__Source of figure__: James, Gareth, et al. An introduction to statistical learning. Vol. 112. New York: springer, 2013.
![Source: James, Gareth, et al. An introduction to statistical learning. Vol. 112. New York: springer, 2013.](image1.png)

We do not want to sacrifice this high accuracy to explainability.  Hence: __ML explainability__. There are a lot of ML explainability tools out there, _in the wild_.

The `teller` is a __model-agnostic tool for ML explainability__. _Agnostic_, as long as the input ML model possesses methods `fit` and `predict`, and is applied to tabular data. The `teller` relies on:

- [Finite differences](https://en.wikipedia.org/wiki/Finite_difference) to explain ML models predictions: a little increase in model's explanatory variables + a little decrease, and we can obtain approximate sensitivities of its predictions to changes in these explanatory variables. 
- [Conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction) (so far, as of october 2022) to obtain prediction intervals for ML regression methods


The __teller__'s source code is [available on GitHub](https://github.com/Techtonique/teller), and you can read posts about it [in this blog](https://thierrymoudiki.github.io/blog/#ExplainableML).

Looking for a specific function? You can also use the __search__ function available in the navigation bar.

## Installing

- From Pypi, stable version:

```bash
pip install the-teller
```

- From Github, for the development version: 

```bash
pip install git+https://github.com/Techtonique/teller.git
```

## Quickstart 

- [Heterogeneity of marginal effects](https://github.com/Techtonique/teller/tree/master/teller/demo/thierrymoudiki_011119_boston_housing.ipynb)

- [Significance of marginal effects](https://github.com/Techtonique/teller/tree/master/teller/demo/thierrymoudiki_081119_boston_housing.ipynb)

- [Model comparison](https://github.com/Techtonique/teller/tree/master/teller/demo/thierrymoudiki_151119_boston_housing.ipynb)

- [Classification](https://github.com/Techtonique/teller/tree/master/teller/demo/thierrymoudiki_041219_breast_cancer_classif.ipynb)

- [Interactions](https://github.com/Techtonique/teller/tree/master/teller/demo/thierrymoudiki_041219_boston_housing_interactions.ipynb)

- [Prediction intervals for regression](https://github.com/Techtonique/teller/tree/master/teller/demo/thierrymoudiki_031022_diabetes_pred_interval.ipynb)

## Documentation

- For the [Explainer](documentation/explainer.md)

- For the [Comparator](documentation/comparator.md)

- For [PredictionInterval](documentation/prediction_interval.md)

## References

For **sensitivity analysis**: 

- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (1992). Numerical recipes in C (Vol. 2). Cambridge: Cambridge university press.
- Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 2019-01-04]
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

For **prediction intervals**: 

- Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. Advances in neural information processing systems, 32.

## Contributing

Want to contribute to __teller__'s development on Github, [read this](CONTRIBUTING.md)!

<script async defer src="https://buttons.github.io/buttons.js"></script>