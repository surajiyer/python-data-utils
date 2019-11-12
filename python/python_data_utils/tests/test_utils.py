import pandas as pd
import os
from .. import utils as pyu


def test_load_dataset_1():
    # test basic loading of existing file from cache directory
    data = pyu.load_dataset('dutch_dictionary_small',
                            os.path.join(__file__, '..', '..'))
    assert isinstance(data, pd.DataFrame)
    assert data.columns.tolist() == ['word', 'count']
    assert data.shape == (50000, 2)


# def test_load_dataset_2():
#     # test creation of default cache directory
#     data = pyu.load_dataset('dutch_dictionary_small')
#     assert os.path.isdir(pyu.DEFAULT_CACHE_DIR)
#     assert isinstance(data, pd.DataFrame)
#     assert data.columns.tolist() == ['word', 'count']
#     assert data.shape == (50000, 2)


def test_get_catalog():
    assert isinstance(pyu.get_catalog(), dict)


def test_generate_random_string():
    string = pyu.generate_random_string()
    assert len(string) == 10
    assert all(c.isupper() or c.isdigit() for c in string)
