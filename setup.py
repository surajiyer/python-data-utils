from setuptools import setup, find_packages
try:
    import builtins
except ImportError:
    # Python 2 compat: just to be able to declare that Python >=3.5 is needed.
    import __builtin__ as builtins
builtins.__PYDU_SETUP__ = True

# We can actually import a restricted version of python_data_utils that
# does not need the compiled code
from python.python_data_utils import __version__


install_requires = [
    'pandas',
    'numpy',
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
    'statsmodels',
    'openpyxl',
    'pyyaml'
]
test_requirements = ['pytest']

setup(
    name='python_data_utils',
    package_dir={'': 'python'},
    version=__version__,
    author='Suraj Iyer',
    author_email='iyer.suraj@outlook.com',
    description='🚀 Utility classes and functions for common data science libraries',
    long_description=open('README.md', encoding='utf-8').read(),
    url='https://github.com/surajiyer/python-data-utils',
    license='Apache 2.0',
    packages=find_packages('python', exclude=('tests*',)),  # include all packages under 'python'
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
