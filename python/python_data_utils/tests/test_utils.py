import pandas as pd
import os
import shutil
from .. import utils as pyu
import importlib.resources as pkg_resources


def test_load_dataset_1():
    # test basic loading of existing file from cache directory
    with pkg_resources.path('python_data_utils', '__init__.py') as p:
        path = os.path.split(p)[0]
    data = pyu.load_artifact('dutch_dictionary_small', path)
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == ['word', 'count']
    assert data.shape == (50000, 2)


def test_get_catalog():
    # test getting the catalog
    assert isinstance(pyu.get_catalog(), dict)


def test_load_dataset_2():
    # test creation of default cache directory
    # test loading from url
    name = 'dutch_dictionary_small'
    data = pyu.load_artifact(name)
    assert os.path.isdir(pyu._DEFAULT_CACHE_DIR)
    assert os.path.isfile(pyu.fix_path(os.path.join(
        pyu._DEFAULT_CACHE_DIR,
        pyu.get_catalog()[name]['filepath'])))
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == ['word', 'count']
    assert data.shape == (50000, 2)
    shutil.rmtree(pyu._DEFAULT_CACHE_DIR, ignore_errors=True)


def test_generate_random_string():
    # test generation of random string with default parameters
    string = pyu.generate_random_string()
    assert len(string) == 10
    assert all(c.isupper() or c.isdigit() for c in string)
