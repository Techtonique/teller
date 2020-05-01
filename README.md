![teller logo](the-teller.png)

<hr>  

![PyPI](https://img.shields.io/pypi/v/teller) [![PyPI - License](https://img.shields.io/pypi/l/the-teller)](https://github.com/thierrymoudiki/teller/blob/master/LICENSE) [![Downloads](https://pepy.tech/badge/the-teller)](https://pepy.tech/project/the-teller) [![Documentation Status](https://readthedocs.org/projects/the-teller/badge/?version=latest)](https://the-teller.readthedocs.io/en/latest/?badge=latest)


There is an increasing need for __transparency__ and __fairness__ in Machine Learning (ML) models  predictions. Consider for example a banker who has to explain to a client why his/her loan application is rejected, or a health professional who must explain what constitutes his/her diagnosis. Some ML models are indeed very accurate, but are considered  hard to explain, relatively to popular linear models. 


__Source of figure__: James, Gareth, et al. An introduction to statistical learning. Vol. 112. New York: springer, 2013.
![Source: James, Gareth, et al. An introduction to statistical learning. Vol. 112. New York: springer, 2013.](image1.png)

We do not want to sacrifice this high accuracy to explainability.  Hence: __ML explainability__. There are a lot of ML explainability tools out there, _in the wild_ (don't take my word for it).

The `teller` is a __model-agnostic tool for ML explainability__ - agnostic, as long as  this model possesses methods `fit` and `predict`. The `teller`'s philosophy is to rely on [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) to explain ML models predictions: a little increase in model's explanatory variables + a little decrease, and we can obtain approximate sensitivities of its predictions to changes in these explanatory variables. 


## Installation 

- From Pypi, stable version:

```bash
pip install the-teller
```

- From Github, for the development version: 

```bash
pip install git+https://github.com/thierrymoudiki/teller.git
```


## Package description

These notebooks will be some good introductions:

- [Heterogeneity of marginal effects](/teller/demo/thierrymoudiki_011119_boston_housing.ipynb)
- [Significance of marginal effects](/teller/demo/thierrymoudiki_081119_boston_housing.ipynb)
- [Model comparison](/teller/demo/thierrymoudiki_151119_boston_housing.ipynb)
- [Classification](/teller/demo/thierrymoudiki_041219_breast_cancer_classif.ipynb)
- [Interactions](/teller/demo/thierrymoudiki_041219_boston_housing_interactions.ipynb)


## Contributing

Your contributions are welcome, and valuable. Please, make sure to __read__ the [Code of Conduct](CONTRIBUTING.md) first. 

If you're not comfortable with Git/Version Control yet, please use [this form](https://forms.gle/Y18xaEHL78Fvci7r8).

In Pull Requests, let's strive to use [`black`](https://black.readthedocs.io/en/stable/) for formatting: 

```bash
pip install black
black --line-length=80 file_submitted_for_pr.py
```


## Dependencies 

- Numpy
- Pandas
- Scipy
- scikit-learn


## Citation

```
@misc{moudiki2019teller,
	author={Moudiki, T.},
	title={\code{teller}, {M}odel-agnostic {M}achine {L}earning explainability},
	howpublished={\url{https://github.com/thierrymoudiki/teller}},
	note={BSD 3-Clause Clear License. Version 0.x.x.},
	year={2019--2020}
	}
```


## References

- Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P. (1992). Numerical recipes in C (Vol. 2). Cambridge: Cambridge university press.
- Jones E, Oliphant E, Peterson P, et al. SciPy: Open Source Scientific Tools for Python, 2001-, http://www.scipy.org/ [Online; accessed 2019-01-04]
- Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.


## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2019. 

