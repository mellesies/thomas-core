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

from .factor.factor import Factor
from .factor.cpt import CPT
from .factor.node import Node, DiscreteNetworkNode

from .collection.bag import Bag
from .collection.bayesiannetwork import BayesianNetwork

from . import error as e

import logging
log = logging.getLogger('pybn')

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


def enable_logging_to_console(enable=True):
    format_ = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if enable:
        logging.basicConfig(format=format_, level=logging.DEBUG)
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.CRITICAL)

def _process_prefix_index(idx, add_or_remove):
    """Add/remove prefixes to/from an Index or MultiIndex."""
    if not isinstance(idx, (pd.Index, pd.MultiIndex)):
        raise TypeError('index should be pandas.Index or pandas.MultiIndex')

    if isinstance(idx, pd.MultiIndex):
        # processed_states will keep track of a list of tuples:
        #   [('i0', ), ('d0', 'd1')]
        processed_states = []

        # MultiIndex.from_product requires names to be iterable, so we'll need
        # to cast it anyway.
        names = list(idx.names)

        # idx.levels[i] only takes integers as index, so we'll have to enumerate
        for i, name in enumerate(names):
            prefix = name + '.'

            # MultiIndex keeps its levels sorted, so the order we're seeing in
            # idx.levels[i] is not necessarily the order that was originally
            # defined. We'll use idx.codes later to set this right.
            if add_or_remove == 'add':
                new_states = [prefix + s for s in idx.levels[i]]
            else:
                new_states = [s.replace(prefix, '') for s in idx.levels[i]]

            processed_states.append(new_states)


        # Creating a new MultiIndex from product uses the order as specified
        # in the tuples. Since this may not have been the original order, we'll
        # need to set this straight by calling `new_idx.set_codes()`.
        new_idx = pd.MultiIndex.from_product(processed_states, names=names)
        new_idx = new_idx.set_codes(idx.codes)
        return new_idx

    # Regular Index
    prefix = idx.name + '.'

    if add_or_remove == 'add':
        new_states = [prefix + state for state in idx]
    else:
        new_states = [state.replace(prefix, '') for state in idx]

    # For some reason the regular Index does not sort its states/levels, so
    # there's no need to call `set_codes()`.
    return pd.Index(new_states, name=idx.name)

def add_prefix_to_index(idx):
    """Add prefixes to an Index or MultiIndex."""

    # It seems the entire prefix-thing is completely unnecessary
    # return _process_prefix_index(idx, 'add')
    return idx

def add_prefix_to_dict(variable_states):
    """Add prefixes to dict with states."""

    # It seems the entire prefix-thing is completely unnecessary
    return variable_states

    # prefixed = {}
    #
    # for name, states in variable_states.items():
    #     prefix = f'{name}.'
    #     if isinstance(states, str):
    #         if states.startswith(prefix):
    #             prefixed[name] = f'{states}'
    #         else:
    #             prefixed[name] = f'{prefix}{states}'
    #     else:
    #         prefixed[name] = [f'{prefix}{s}' if not s.startswith(prefix) else s for s in states]
    #
    # return prefixed

def remove_prefix_from_index(idx):
    """Remove any prefixes from a dict with variable states."""
    # It seems the entire prefix-thing is completely unnecessary
    return idx
    # return _process_prefix_index(idx, 'remove')

def remove_from_dict_by_value(dict_, value):
    """filter entries that have value `value` from the dict."""
    return {k:v for k,v in evidence_values.items() if v == value}

def remove_none_values_from_dict(dict_):
    """Remove none values, like `None` and `np.nan` from the dict."""
    t = lambda x: (x is None) or (isinstance(x, float) and np.isnan(x))
    result = {k:v for k,v in dict_.items() if not t(v)}
    return result

def index_to_dict(idx):
    if isinstance(idx, pd.MultiIndex):
        return {i.name: list(i) for i in idx.levels}

    return {idx.name: list(idx)}

