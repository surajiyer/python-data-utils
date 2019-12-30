# coding: utf-8

"""
    description: Python data utility functions and classes
    author: Suraj Iyer
"""

import sys

__version__ = '0.1.0'

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to enable importing subpackages of sklearn when
    # the binaries are not built
    __PYDU_SETUP__
except NameError:
    __PYDU_SETUP__ = False

if __PYDU_SETUP__:
    sys.stderr.write('Partial import of python_data_utils during the build process.\n')
else:
    from . import decorator
