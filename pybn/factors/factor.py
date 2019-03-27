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

import pybn
from .. import error as e


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

            # Prefix the states of the variables to make them unique.
            # {'I': ['i0', 'i1']} --> {'I': ['I.i0', 'I.i1']}
            idx = pybn.add_prefix_to_index(idx)

        elif (isinstance(data, pd.Series) 
            and isinstance(data.index, (pd.Index, pd.MultiIndex))):
                # TODO: should we make sure that the index is prefixed?
                data = data.copy()
                idx = data.index

        elif isinstance(data, Factor):
            data = data._data.copy()
            idx = data._data.index

        else:
            msg =  'data should either be a pandas.Series *with* a proper'
            msg += ' index or variable_states should be provided.'
            print('Oh dear: ', type(data))
            print('variable_states: ', variable_states)
            raise Exception(msg)

        # Set self._data
        self._data = pd.Series(data, index=idx)

    def __repr__(self):
        """repr(f) <==> f.__repr__()"""
        return f'{self.display_name}\n{repr(self._data_without_prefix)}'

    def __mul__(self, other):
        """A * B <=> A.mul(B)"""
        return self.mul(other)

    def __truediv__(self, other):
        """A / B <=> A.div(B)"""
        return self.div(other)

    def __getitem__(self, key):
        """A[x] <==> A.__getitem__(x)"""
        scope = self.scope

        if isinstance(key, str):
            key = f'{scope[0]}.{key}'

        elif isinstance(key, tuple):
            prefixed = []
            for i, k in enumerate(key):
               prefixed.append(f'{scope[i]}.{k}')

            key = tuple(prefixed)

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
        return self._data_without_prefix.index
    
    @property
    def values(self):
        """Return the factor values as an np.array"""
        return self._data.values
    
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

    @property
    def _data_without_prefix(self):
        """Return a copy of the underlying data where the states have *not* been
            prefixed with the random variable's name.
        """
        data = self._data.copy()

        if isinstance(data.index, pd.MultiIndex):
            idx = data.index.remove_unused_levels()
        else:
            idx = data.index

        try:
            idx = pybn.remove_prefix_from_index(idx)
            data.index = idx
            return data
        except Exception as e:
            print()
            print('Exception:', e)
            print(idx)
            raise

    def max(self):
        """Proxy for pandas.Series.max()"""
        return self._data_without_prefix.max()

    def idxmax(self):
        """Proxy for pandas.Series.idmax()"""
        return self._data_without_prefix.idxmax()

    def sum(self):
        """Sum all values of the factor."""
        return self._data.sum()

    def mul(self, other, *args, **kwargs):
        """A * B <=> A.mul(B)"""
        if isinstance(other, Factor):
            if (not self.overlaps_with(other)
                and self.display_name != other.display_name):
                return self.outer(other)

            # Other is a factor with overlap with `self`
            other = other.reorder_scope()

        # Safety precaution. Really only necessary when multiplying a Series or
        # another Factor.
        me = self.reorder_scope()

        if len(me) == 1 and len(other) == 1:
            # Pandas has the nasty habit to mess up multiindexes when 
            # multiplying two Series with a single row. At that point, order
            # suddenly becomes important (i.e. s1 * s2 != s2 * s1) and the index
            # of the first Series is reused. The code below is to make sure
            # that all levels of the index are copied to the end result.
            multiplied = me._data.mul(other._data, *args, **kwargs)
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

        return Factor(me._data.mul(other._data, *args, **kwargs))

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
        other_scope = set(other.scope)
        result = len(own_scope.intersection(other_scope)) > 0

        return result

    @property
    def display_name(self):
        names = [n for n in self._data.index.names if n is not None]
        names = ','.join(names)

        if not names:
            names = '?'

        return f'factor({names})'

    @property
    def scope(self):
        """Return the scope of this factor."""
        return [v for v in self._data.index.names]

    @property
    def width(self):
        """Return the width of this factor."""
        return len(self.scope)

    @property
    def variable_states(self):
        """Return a dict of variable states."""
        index = self._data_without_prefix.index
        return pybn.index_to_dict(index)

    def reorder_scope(self, order=None):
        """Reorder the variables in the scope."""
        if self.width > 1:
            if order is None:
                order = self.scope
                order.sort()

            data = self._data.reorder_levels(order)
        else:
            data = self._data.copy()

        return Factor(data)

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

        scope = set(self.scope)

        if not variable_set.issubset(scope):
            raise e.NotInScopeError(variable_set, scope)

        if len(variable_set) == self.width:
            return self.sum()

        # Unstack the requested variables into columns and sum over them.
        unstacked = self._data.unstack(variable)
        summed = unstacked.sum(axis=1)

        return Factor(summed)

    def stack(self, *args, **kwargs):
        """Proxy for pd.Series.stack()."""
        return self._data.stack(*args, **kwargs)

    def unstack(self, *args, **kwargs):
        """Proxy for pd.Series.unstack()."""
        # FIXME: unstack leaks index prefixes ... 
        return self._data.unstack(*args, **kwargs)

    def dot(self, other):
        """Return the dot (matrix) product."""
        if isinstance(other, Factor):
            return Factor(self._data.dot(other._data))

        # Hail mary ... ;)
        return Factor(self._data.dot(other))

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
        except:
            print('Could not create Factor from outer product?')
            print('type(stacked):', type(stacked))
            print('stacked:', stacked)
            raise

        return f

    def droplevel(self, level):
        """Proxy for pandas.Series.droplevel()"""
        return Factor(self._data.droplevel(level))

    def set_evidence(self, **kwargs):
        """Return a reduced factor."""

        # when called like set_evidence(D='d1', E='e0'), we'll need to 
        # set rows that do not correspond to the evidence to zero.

        kwargs = pybn.add_prefix_to_dict(kwargs)

        # Find the (subset of) evidence that's related to this factor's scope.
        levels = [l for l in kwargs.keys() if l in self.scope]
        data = self._data.copy()

        for l in levels:
            value = kwargs[l]

            if value not in data.index.get_level_values(l):
                value = value.replace(f'{l}.', '')
                raise e.InvalidStateError(l, value, self)

            idx = data.index.get_level_values(l) != value
            data[idx] = np.nan

        # data = data.dropna()
        #        
        # if len(data) == 0:
        #     raise Exception('this is weird!?')

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
        idx = pybn.remove_prefix_from_index(data.index)
        variable_states = pybn.index_to_dict(idx)

        return {
            'type': 'Factor',
            'scope': self.scope,
            'variable_states': variable_states,
            'data': data.to_list(),
        }

    @classmethod
    def from_dict(cls, d):
        """Return a Factor initialized by its dict representation."""
        return Factor(d['data'], d['variable_states'])


