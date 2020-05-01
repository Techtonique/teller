from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.5.0'


with open("README.md", "r") as fh:
    long_description = fh.read()

# get the dependencies and installs
here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(
    name='the-teller',
    version=__version__,
    description='teller, explainable machine learning',
    long_description='Model-agnostic Machine Learning explainability',
    url='https://github.com/thierrymoudiki/teller',
    alias='the-teller',
    download_url='https://github.com/thierrymoudiki/teller/tarball/' + __version__,
    license='BSD',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests*']),
    include_package_data=True,
    author='T. Moudiki',
    install_requires=["numpy >= 1.13.0", "pandas >= 0.25.1", 
                      "scipy >= 0.19.0", "scikit-learn >= 0.18.0"].append(install_requires),
    author_email='thierry.moudiki@gmail.com'
)
