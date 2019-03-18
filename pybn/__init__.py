# -*- coding: utf-8 -*-
"""Module with classes that facilitate building a Bayesian network.

Examples are borrowed from Koller and Friedmand's "Probabilistic
Graphical Models: Principles and Techniques"
"""
import os
from datetime import datetime as dt

from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

from .factors.factor import Factor
from .factors.cpt import CPT
from .factors.node import Node

from .collections.bag import Bag
from .collections.bayesiannetwork import BayesianNetwork

from . import error as e

def find_last_modified_script():
    directory = os.path.dirname(__file__)
    all_files = []

    for root, dirs, files in os.walk(directory):
        files = [os.path.join(root, f) for f in files if f.endswith('.py')]

        all_files += files

    fmt = "%Y-%m-%d %H:%M"
    all_mtimes = [os.path.getmtime(f) for f in all_files]
    all_mtimes_hr = [dt.fromtimestamp(t).strftime(fmt) for t in all_mtimes]

    as_tuples = zip(all_files, all_mtimes, all_mtimes_hr)
    as_tuples = sorted(as_tuples, key=lambda x: x[1], reverse=True)

    return as_tuples[0][0], as_tuples[0][2]


version = find_last_modified_script()
__version_long__ = '{} ({})'.format(version[1], version[0])
__version__ = find_last_modified_script()[1]
del version


