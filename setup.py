from pathlib import Path
from setuptools import setup, find_packages

package_dir = 'python_data_utils'
root = Path(__file__).parent.resolve()

# Read in package meta from about.py
about_path = root / package_dir / 'about.py'
with about_path.open('r', encoding='utf8') as f:
    about = {}
    exec(f.read(), about)

# Get readme
readme_path = root / 'README.md'
with readme_path.open('r', encoding='utf8') as f:
    readme = f.read()

install_requires = [
    'beautifulsoup4', 'dill', 'distance', 'fake-useragent',
    'fuzzywuzzy', 'hdbscan', 'ipython', 'ipywidgets', 'joblib',
    'matplotlib', 'nltk', 'numba', 'numpy', 'openpyxl',
    'pandas', 'parse', 'progressbar2', 'pyppeteer>=0.0.14',
    'pyquery', 'pyyaml', 'requests', 'scikit-learn', 'scipy',
    'seaborn', 'urlclustering', 'w3lib'
]
test_requires = ['pytest']
extras_require = {
    'spark': ['pyspark>=2.4.0,<3.0.0'],
    'statsmodels': ['statsmodels']
}

setup(
    name=about['__title__'],
    description=about['__summary__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    author=about['__author__'],
    author_email=about['__email__'],
    url=about['__uri__'],
    version=about['__version__'],
    license=about['__license__'],
    packages=find_packages(exclude=('tests*',)),
    install_requires=install_requires,
    test_requires=test_requires,
    extras_require=extras_require,
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
