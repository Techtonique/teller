from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.5.0'

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

setup(
    name='teller',
    version=__version__,
    description='teller',
    long_description=long_description,
    url='https://github.com/thierrymoudiki/teller',
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
    dependency_links=dependency_links,
    author_email='thierry.moudiki@gmail.com'
)
