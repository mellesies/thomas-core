# -*- coding: utf-8 -*-
"""Factor: the basis for all reasoning."""
import os
from datetime import datetime as dt

from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

import logging
log = logging.getLogger('thomas')

import thomas.core
import thomas.core.base
from thomas.core import error

# ------------------------------------------------------------------------------
# Helper functions.
# ------------------------------------------------------------------------------
"""
def mul(x1, x2, debug=False):
    if debug:
        try:
            print('-' * 80)
            print(f'x1: {x1.scope}')
            print(x1)
            print('-' * 80)
            print(f'x2: {x2.scope}')
            print(x2)
            print()
        except Exception as e:
            print('x1:', x1, type(x1))
            print('-' * 80)
            print('x2:', x2, type(x2))
            print()

    try:
        if isinstance(x1, Factor):
            result = x1.mul(x2)
        elif isinstance(x2, Factor):
            result = x2.mul(x1)
        else:
            result = x1 * x2

    except Exception as e:
        print('-' * 80)
        print(e)
        print('-' * 80)
        print('could not multiply two factors')
        print(f'x1: {x1.display_name}: {x1.scope}')
        print(x1)
        print('-' * 80)
        print(f'x2: {x2.display_name}: {x2.scope}')
        print(x2)
        print()
        raise

    if debug:
        try:
            print('result:', result.scope)
        except Exception as e:
            print('result is not a factor')

        print(result)
        print('-' * 80)
        print()

    if isinstance(result, Factor):
        result = result.reorder_scope()
    return result
"""


def mul(x1, x2):
    """Multiply two Factors with each other.

    Helper function for functools.reduce().
    """
    if isinstance(x1, Factor):
        result = x1.mul(x2)

    elif isinstance(x2, Factor):
        result = x2.mul(x1)

    else:
        result = x1 * x2

    if isinstance(result, Factor):
        result = result.reorder_scope()

    return result


# ------------------------------------------------------------------------------
# Factor
# ------------------------------------------------------------------------------
class Factor(object):
    """Factor."""

    def __init__(self, data, variable_states=None):
        """Initialize a new Factor.

        Args:
            data (list, pandas.Series, Factor): array of values.
            variable_states (dict): list of allowed states for each random
                variable, indexed by name. If variable_states is None, `data`
                should be a pandas.Series with a proper Index/MultiIndex.
        """
        if variable_states:
            # Create a (Multi)Index from the variable states.
            idx = self._index_from_variable_states(variable_states)

        elif (isinstance(data, pd.Series)
            and isinstance(data.index, (pd.Index, pd.MultiIndex))):
                data = data.copy()
                idx = data.index
                variable_states = thomas.core.base.index_to_dict(idx)

        elif isinstance(data, Factor):
            variable_states = data._variable_states
            data = data._data.copy()
            idx = data._data.index

        else:
            msg =  'data should either be a pandas.Series *with* a proper'
            msg += ' index or variable_states should be provided.'
            raise Exception(msg)

        if np.issubdtype(type(data), np.integer):
            data = float(data)

        self._data = pd.Series(data, index=idx).dropna()
        self._variable_states = variable_states

    def __repr__(self):
        """repr(f) <==> f.__repr__()"""
        return f'{self.display_name}\n{repr(self._data)}'

    def __add__(self, other):
        """A + B <=> A.__add__(B)"""
        return self.add(other)

    # def __radd__(self, other):
    #     """A + B <=> A.__radd__(B)"""
    #     return self.add(other)

    def __mul__(self, other):
        """A * B <=> A.mul(B)"""
        return self.mul(other)

    # def __rmul__(self, other):
    #     """A * B <=> A.mul(B)"""
    #     return self.mul(other)

    def __truediv__(self, other):
        """A / B <=> A.div(B)"""
        return self.div(other)

    def __getitem__(self, key):
        """factor[x] <==> factor.__getitem__(x)"""
        result = self._data.__getitem__(key)

        if isinstance(result, pd.Series):
            return Factor(result)

        return result

    def __len__(self):
        """len(f) == f.__len__()"""
        return len(self._data)

    @property
    def index(self):
        """Return the Factor index (without prefixes)."""
        return self._data.index

    @property
    def values(self):
        """Return the factor values as an np.array"""
        return self._data.values

    @property
    def display_name(self):
        names = [n for n in self._data.index.names if n is not None]
        names = ','.join(names)

        return f'factor({names})'

    @property
    def vars(self):
        """Return the variables in this factor (i.e. the scope) as a *set*."""
        return set(self.scope)

    @property
    def scope(self):
        """Return the scope of this factor."""
        return list(self._data.index.names)

    @property
    def width(self):
        """Return the width of this factor."""
        return len(self.scope)

    @property
    def variable_states(self):
        """Return a dict of variable states."""
        if self._variable_states is not None:
            return self._variable_states

    @classmethod
    def _index_from_variable_states(cls, variable_states):
        """Create an pandas.Index or pandas.MultiIndex from a dictionary."""
        if len(variable_states) == 1:
            # Cast type dict_items to a list so we can use indexes
            variable_states = list(variable_states.items())

            # The first and only item is a tuple
            var_name, var_states = variable_states[0]

            # Create a pandas.Index
            idx = pd.Index(var_states, name=var_name)

        else:
            idx = pd.MultiIndex.from_product(
                variable_states.values(),
                names=variable_states.keys()
            )

        return idx

    def equals(self, other, precision=3):
        """Test whether two objects contain the same elements."""
        return self._data.round(precision).equals(other._data.round(precision))

    def max(self):
        """Proxy for pandas.Series.max()"""
        return self._data.max()

    def idxmax(self):
        """Proxy for pandas.Series.idmax()"""
        return self._data.idxmax()

    def sum(self):
        """Sum all values of the factor."""
        return self._data.sum()

    def add(self, other):
        """A + B <=> A.__add__(B)"""
        if isinstance(other, (pd.Series, int, float, np.float, np.float64)):
            # Add the data as pd.Series
            f2 = self._data.add(other)
            f2[f2.isna()] = 0
            return Factor(f2)

        elif isinstance(other, Factor):
            return Factor(self._data.add(other._data))

        raise Exception(f'Unsure how to add {type(other)}')

    def mul(self, other):
        """A * B <=> A.mul(B)"""
        if isinstance(other, (int, float, np.float, np.float64)):
            # Handle multiplication with scalars quick and easy.
            return Factor(self._data.mul(other), self.variable_states)

        elif isinstance(other, Factor):
            if (not self.overlaps_with(other)
                and self.display_name != other.display_name):
                return self.outer(other)

            # If we're here, other is a Factor *with* overlap with `self`
            other = other.reorder_scope()

        elif isinstance(other, pd.Series):
            # This only works if the series has a proper (multi)index set.
            other = Factor(other)


        # Safety precaution. Really only necessary when multiplying a Series or
        # another Factor.
        me = self.reorder_scope()

        if len(me) == 1 and len(other) == 1:
            # Pandas has the nasty habit to mess up multiindexes when
            # multiplying two Series with a single row. At that point, order
            # suddenly becomes important (i.e. s1 * s2 != s2 * s1) and the index
            # of the first Series is reused. The code below is to make sure
            # that all levels of the index are copied to the end result.
            multiplied = me._data.mul(other._data)
            i1, i2 = multiplied.index, other._data.index
            names = [n for n in i2.names if n not in i1.names]

            if not names:
                # Apparently all levels of the index made it to the result.
                # We're done.
                return Factor(multiplied)

            # Apparently we're missing one or more indices. We know that we're
            # dealing with an index for a single row, so we can safely get the
            # first item from the level values
            keys = [i2.get_level_values(n)[0] for n in names]

            # We're creating an index for a single row: tuple(keys) is the
            # actual index.
            keys = [tuple(keys)]
            concatted = pd.concat([multiplied], keys=keys, names=names)
            return Factor(concatted)

        result = me._data.mul(other._data)
        return Factor(result)

    def div(self, other, *args, **kwargs):
        """A / B <=> A.mul(B)"""
        if isinstance(other, Factor):
            return Factor(self._data.div(other._data, *args, **kwargs))

        return Factor(self._data.div(other))

    def overlaps_with(self, other):
        """Return True iff the scope of this Factor overlaps with the scope of
        other.
        """
        own_scope = set(self.scope)

        if isinstance(other, Factor):
            other_scope = set(other.scope)
        else:
            other_scope = set(other)

        return len(own_scope.intersection(other_scope)) > 0

    def reorder_scope(self, order=None):
        """Reorder the variables in the scope."""
        if self.width > 1:
            if order is None:
                order = self.scope
                order.sort()

            data = self._data.reorder_levels(order)
            data = data.sort_index()

        else:
            data = self._data

        return Factor(data)

    def project(self, Q):
        """Project the current factor on Q.

        Args:
            Q (set or str): variable(s) to compute marginal over.

        Returns:
            Factor:  marginal distribution over the RVs in Q.
        """
        if isinstance(Q, (list, tuple)):
            Q = set(Q)

        assert isinstance(Q, (set, str)), "Q should be a set or a string!"

        if isinstance(Q, str):
            Q = {Q}

        vars_to_sum_out = list(set(self.scope) - Q)

        if len(vars_to_sum_out) == 0:
            return Factor(self)

        if self.width == 1:
            return self

        return self.sum_out(vars_to_sum_out)

    def sum_out(self, variable):
        """Sum-out a variable (or list of variables) from the factor.

        This essentially removes a variable from the distribution.

        Args:
            variable (str, list): Name or list of names of variables to sum out.

        Returns:
            Factor: factor with the specified variable removed.
        """
        if isinstance(variable, (str, tuple)):
            variable_set = set([variable])
        else:
            variable_set = set(variable)

        if len(variable_set) == 0:
            # Nothing to sum out ...
            return Factor(self)

        scope = set(self.scope)

        if not variable_set.issubset(scope):
            raise error.NotInScopeError(variable_set, scope)

        if len(variable_set) == self.width:
            return self.sum()

        # Unstack the requested variables into columns and sum over them.
        unstacked = self._data.unstack(tuple(variable_set))
        summed = unstacked.sum(axis=1)
        return Factor(summed)

    def unstack(self, *args, **kwargs):
        """Proxy for pd.Series.unstack()."""
        return self._data.unstack(*args, **kwargs)

    # melle: This wasn't used in any of the real code, only in the notebooks
    #        as example.
    # def dot(self, other):
    #     """Return the dot (matrix) product."""
    #     if isinstance(other, Factor):
    #         # return Factor(self._data.dot(other._data))
    #         other = other._data.unstack()

    #     return Factor(self._data.dot(other))

    def outer(self, other):
        """Return the outer product."""
        df = pd.DataFrame(
            np.outer(self._data, other._data),
            index=self._data.index,
            columns=other._data.index
        )

        if isinstance(df.columns, pd.MultiIndex):
            levels = df.columns.levels
            stacked = df.stack(list(range(len(levels)))).squeeze()
        else:
            stacked = df.stack().squeeze()

        try:
            f = Factor(stacked)

        except Exception as e:
            log.error('Could not create Factor from outer product?')
            log.error(f'type(stacked): {type(stacked)}')
            log.error(f'stacked: {stacked}')
            log.exception(e)
            raise

        return f

    def droplevel(self, level):
        """Proxy for pandas.Series.droplevel()"""
        return Factor(self._data.droplevel(level))

    def extract_values(self, **kwargs):
        """Extract entries from the Factor by RV and value.

        Kwargs:
            dict of states, indexed by RV. E.g. {'G': 'g1'}
        """
        # Need to do some trickery to get the correct levels from the
        # MultiIndex.
        indices = []

        for level, value in kwargs.items():
            idx = self._data.index.get_level_values(level) == value
            indices.append(list(idx))

        zipped = list(zip(*indices))
        idx = [all(x) for x in zipped]
        data = self._data[idx]

        levels = list(kwargs.keys())
        last_level = levels[-1]
        last_value = kwargs[last_level]

        levels = levels[:-1]
        data.index = data.index.droplevel(levels)

        # Testing for pd.Index doesn't work since isinstance() caters for
        # inheritance
        if isinstance(data.index, pd.MultiIndex):
            return Factor(data)

        # This should be a scalar ...
        return data[last_value]

    def keep_values(self, **kwargs):
        """Return a reduced factor."""

        # when called like set_evidence(D='d1', E='e0'), we'll need to
        # set rows that do not correspond to the evidence to zero.

        # Find the (subset of) evidence that's related to this factor's scope.
        levels = [l for l in kwargs.keys() if l in self.scope]
        data = self._data.copy()

        for l in levels:
            value = kwargs[l]

            if value not in data.index.get_level_values(l):
                value = value.replace(f'{l}.', '')
                raise error.InvalidStateError(l, value, self)

            idx = data.index.get_level_values(l) != value
            data[idx] = np.nan

        return Factor(data.dropna())

    def normalize(self):
        """Normalize the factor so the sum of all values is 1."""
        return self.__class__(self._data / self._data.sum())

    def sort_index(self, *args, **kwargs):
        """Sort the index of the Factor."""
        return Factor(self._data.sort_index(*args, **kwargs))

    def as_series(self):
        """Return the Factor as a pandas.Series."""
        return pd.Series(self._data)

    def as_dict(self):
        """Return a dict representation of this Factor."""
        data = self._data

        if self.width > 1:
            data = data.reorder_levels(self.scope)

        # Remove any prefixes ...
        variable_states = self.variable_states

        return {
            'type': 'Factor',
            'scope': self.scope,
            'variable_states': variable_states,
            'data': data.to_list(),
        }

    def zipped(self):
        """Return a dict with data, indexed by tuples."""
        index = list(self.index)
        return dict(zip(index, self._data.to_list()))

    @classmethod
    def from_dict(cls, d):
        """Return a Factor initialized by its dict representation."""
        return Factor(d['data'], d['variable_states'])


