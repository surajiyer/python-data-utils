# coding: utf-8

"""
    description: General utility functions / methods
    author: Suraj Iyer
"""

__all__ = [
    'load_from_url',
    'load_artifact',
    'clear_cache',
    'get_catalog',
    'generate_random_string'
]

import yaml
import urllib.request
import progressbar as pb
import dill
import shutil
import pandas as pd
from typing import Any
import os
import importlib.resources as pkg_resources


def _load_catalog():
    with pkg_resources.path('python_data_utils.conf', 'catalog.yml') as p:
        with open(p) as stream:
            return yaml.safe_load(stream)


def get_catalog():
    import copy
    return copy.deepcopy(_catalog)


def fix_path(path: str) -> str:
    assert isinstance(path, str)
    return os.path.abspath(os.path.expanduser(path))


# Load default cache dir
with pkg_resources.path('python_data_utils.conf', 'basic.yml') as p:
    with open(p) as stream:
        _DEFAULT_CACHE_DIR = fix_path(
            yaml.safe_load(stream)['default_cache_dir'])

# Load dataset catalog
_catalog = _load_catalog()


def load_from_url(url: str, save_path: str):
    # Create progressbar
    widgets = ['Downloaded: ', pb.Percentage(),
               ' ', pb.Bar(marker=pb.RotatingMarker()),
               ' ', pb.ETA(),
               ' ', pb.FileTransferSpeed()]
    pbar = pb.ProgressBar(widgets=widgets)

    def dl_progress(count, blockSize, totalSize):
        if pbar.max_value is None:
            pbar.max_value = totalSize
            pbar.start()
        pbar.update(min(count * blockSize, totalSize))

    # Create the save path if not exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Download the file
    print(f'Downloading file from {url}.')
    urllib.request.urlretrieve(url, save_path, reporthook=dl_progress)

    # Close progressbar
    pbar.finish()


def _load_file(path: str, file_type: str, configs: dict = None) -> Any:
    # open file once cached
    if file_type == 'txt':
        with open(path, 'r', **configs) as f:
            data = f.read()
    elif file_type == "pkl":
        data = dill.load(path, **configs)
    elif file_type == 'csv':
        configs.pop('filepath_or_buffer', None)
        data = pd.read_csv(path, **configs)
    elif file_type == 'xlsx':
        configs.pop('io', None)
        data = pd.read_excel(path, **configs)
    elif file_type == 'parquet':
        configs.pop('path', None)
        data = pd.read_parquet(path, **configs)
    else:
        raise NotImplementedError('Unknown file type.')

    return data


def load_artifact(name: str, cache_dir: str = None) -> Any:
    """
    Load any artifact from the catalog.

    :param name: str
        name of the artifact as defined in the catalog.
    :param cache_dir: str
        Directory to cache it in.
    :return: Any
    """
    catalog = get_catalog()
    assert name in catalog, 'Cannot find given dataset.'
    assert cache_dir is None or isinstance(cache_dir, str)

    # if cache directory not given
    if cache_dir is None:
        # set default cache directory
        cache_dir = _DEFAULT_CACHE_DIR
    else:
        cache_dir = fix_path(cache_dir)

    # if cache directory does not exist
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)

    # load data
    filepath = catalog[name].pop('filepath', None)
    url = catalog[name].pop('url', None)
    filetype = catalog[name].pop('type', None)
    if filepath is None:
        if url is None:
            raise ValueError(f"Dataset '{name}' not available.")
        else:
            filepath = fix_path(os.path.join(cache_dir, os.path.split(url)[1]))
            if not os.path.isfile(filepath):
                load_from_url(url, cache_dir)
    elif not os.path.isfile(os.path.join(cache_dir, filepath)):
        if url is None:
            raise ValueError(f"Dataset '{name}' not available.")
        else:
            filepath = fix_path(os.path.join(cache_dir, filepath))
            load_from_url(url, filepath)
    else:
        filepath = fix_path(os.path.join(cache_dir, filepath))

    return _load_file(filepath, filetype, catalog[name])


def clear_cache(cache_dir: str = None):
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)


def generate_random_string(characters: str = None, N: int = 10) -> str:
    """
    Create randomly generated strings.

    :param characters: str
        character set to sample from.
    :param N: int
        final string length
    :return: str
    """
    import string
    import random
    if characters is None:
        characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(N))
