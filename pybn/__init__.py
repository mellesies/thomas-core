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

def remove_none_values_from_dict(dict_):
   """Remove none values, like `None` and `np.nan` from the dict."""
   t = lambda x: (x is None) or (isinstance(x, float) and np.isnan(x))
   result = {k:v for k,v in dict_.items() if not t(v)}
   return result

def parse_query_string(query_string):
    """Parse a query string into a tuple of query_dist, query_values,
    evidence_dist, evidence_values.

    The query P(I,G=g1|D,L=l0) would imply:
        query_dist = ('I',)
        query_values = {'G': 'g1'}
        evidence_dist = ('D',)
        evidence_values = {'L': 'l0'}
    """
    def split(s):
        dist, values = [], {}
        params = []

        if s:
            params = s.split(',')

        for p in params:
            if '=' in p:
                key, value = p.split('=')
                values[key] = value
            else:
                dist.append(p)

        return dist, values

    query_str, given_str = query_string, ''

    if '|' in query_str:
        query_str, given_str = query_string.split('|')

    return split(query_str) + split(given_str)


class ProbabilisticModel(object):

    def compute_posterior(self, query_dist, query_values, evidence_dist,
        evidence_values, **kwargs):
        """Compute the probability of the query variables given the evidence.

        The query P(I,G=g1|D,L=l0) would imply:
            query_dist = ['I']
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}

        :param tuple query_dist: Random variable to query
        :param dict query_values: Random variable values to query
        :param tuple evidence_dist: Conditioned on evidence
        :param dict evidence_values: Conditioned on values
        :return: pandas.Series (possibly with MultiIndex)
        """
        raise NotImplementedError

    def P(self, query_string):
        """Return the probability as queried by query_string.

        P('I,G=g1|D,L=l0') is equivalent to calling compute_posterior with:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}
        """
        qd, qv, gd, gv = parse_query_string(query_string)
        return self.compute_posterior(qd, qv, gd, gv)

    def MAP(self, query_dist, evidence_values, include_probability=True):
        """Perform a Maximum a Posteriori query."""
        d = self.compute_posterior(query_dist, {}, [], evidence_values)
        evidence_vars = [e for  e in evidence_values.keys() if e in d.scope]

        d = d.droplevel(evidence_vars)

        if include_probability:
            return d.idxmax(), d.max()

        return d.idxmax()


# Convenience imports
from .factor import Factor
from .cpt import CPT
from .bag import Bag
from .bayesiannetwork import BayesianNetwork