from setuptools import setup, find_packages
from python_data_utils import __version__


install_requires = [
    'numpy',
    'pandas',
    'matplotlib',
    'dill',
    'progressbar2',
    'seaborn',
    'ipython',
    'ipywidgets',
    'scikit-learn',
    'nltk',
    'distance',
    'urlclustering',
    'numba',
    'fuzzywuzzy',
    'hdbscan',
    'scipy',
    'beautifulsoup4',
    'statsmodels'
]
test_requirements = ['pytest']


setup(
    name='python_data_utils',
    package_dir={'': './python'},
    version=__version__,
    author='Suraj Iyer',
    author_email='iyer.suraj@outlook.com',
    description='🚀 Utility classes and functions for common data science libraries',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/surajiyer/python-data-utils',
    license='Apache 2.0',
    packages=find_packages('python'),  # include all packages under 'python'
    install_requires=install_requires,
    test_requires=test_requirements,
    zip_safe=True,
    include_package_data=True,
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Utilities',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7'
    ]
)
