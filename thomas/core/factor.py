# -*- coding: utf-8 -*-
"""Factor: the basis for all reasoning."""
import os
from datetime import datetime as dt
import itertools
import warnings

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

import logging
log = logging.getLogger('thomas.factor')

import thomas.core
import thomas.core.base
from thomas.core import error


# ------------------------------------------------------------------------------
# Helper functions.
# ------------------------------------------------------------------------------
def isiterable(obj):
    """Return True iff an object is iterable."""
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True


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

    # if isinstance(result, Factor):
    #     result = result.reorder_scope()

    return result


# ------------------------------------------------------------------------------
# FactorIndex
# ------------------------------------------------------------------------------
class FactorIndex(object):
    """Index for Factors."""

    def __init__(self, states):
        """Initialize a new FactorIndex."""
        self._index = self.get_index_tuples(states)

    def __getitem__(self, RV):
        """index[x] <==> index.__getitem__(x)"""
        pass

    @staticmethod
    def get_index_tuples(states):
        """Return an index as a list of tuples."""
        return list(itertools.product(*states.values()))

# ------------------------------------------------------------------------------
# Factor
# ------------------------------------------------------------------------------
class Factor(object):
    """Factor for discrete variables.

    Code is heavily inspired (not to say partially copied from) by pgmpy's
    DiscreteFactor. See https://github.com/pgmpy/pgmpy/blob/dev/pgmpy/factors/discrete/DiscreteFactor.py
    """

    def __init__(self, data, states):
        """Initialize a new Factor.

        Args:
            data (list): any iterable
            states (dict): dictionary of random variables and their
                corresponding states.
        """
        # msg = f'data ({type(data)}) is not iterable? states: {states}'
        # assert isiterable(data), msg

        msg = f"'states should be a dict, but got {type(states)} instead?"
        msg += f" type(data): {type(data)}"
        assert isinstance(states, dict), msg

        if np.isscalar(data):
            total_size = np.product([len(s) for s in states.values()])
            data = np.repeat(data, total_size)

        # Copy & make sure we're dealing with a numpy array
        data = np.array(data, dtype=float)

        cardinality = [len(i) for i in states.values()]
        expected_size = np.product(cardinality)

        if data.size != expected_size:
            raise ValueError(f"'data' must be of size/length: {expected_size}")

        # copy to prevent modification
        self.states = states.copy()

        # create two dicts of dicts to map variable state names to index numbers
        # and back
        self.name_to_number = {}
        self.number_to_name = {}

        for RV, values in self.states.items():
            self.name_to_number[RV] = {name: nr for nr, name in enumerate(self.states[RV])}
            self.number_to_name[RV] = {nr: name for nr, name in enumerate(self.states[RV])}

        # Storing the data as a multidimensional array helps with addition,
        # subtraction, multiplication and division.
        self.values = data.reshape(cardinality)

    def __repr__(self):
        """repr(f) <==> f.__repr__()"""
        if self.states:
            return f'{self.display_name}\n{repr(self.as_series())}'

        return f'{self.display_name}: {self.values:.2}'

    def __eq__(self, other):
        """f1 == f2 <==> f1.__eq__(f2)"""
        return self.equals(other)

    def __len__(self):
        """len(factor) <==> factor.__len__()"""
        return self.width

    def __getitem__(self, keys):
        """factor[x] <==> factor.__getitem__(x)"""

        if isinstance(keys, list):
            # FIXME: this is probably slow
            indices = [self._keys_to_indices(idx) for idx in keys]
            return [self.values[idx] for idx in indices]

        return self.values[self._keys_to_indices(keys)]

    def __setitem__(self, keys, value):
        """factor[x] = y <==> factor.__setitem__(x, y)"""
        self.values[self._keys_to_indices(keys)] = value

    # def __getattr__(self, key):
    #     if key not in self.states:
    #         msg = f"'{self.__class__.__name__}' object has no attribute '{key}'"
    #         raise AttributeError(msg)
    #     return self.index_for(key)

    def __add__(self, other):
        """A + B <=> A.__add__(B)"""
        return self.add(other)

    def __mul__(self, other):
        """A * B <=> A.mul(B)"""
        return self.mul(other)

    def __truediv__(self, other):
        """A / B <=> A.div(B)"""
        return self.div(other)

    def _keys_to_indices(self, keys):
        """Return the indices for keys ..."""
        if isinstance(keys, (str, slice)):
            keys = (keys, )

        states = [(self.variables[i], state) for i, state in enumerate(keys)]
        indices = [self.get_state_index(RV, state) for RV, state in states]
        return tuple(indices)

    def _get_index_tuples(self):
        """Return an index as a list of tuples.

        Args:
            states (dict): dict of states, indexed by RV

        Return:
            list of tuples, making up all combinations of states.
        """
        # return np.array(list(itertools.product(*states.values())))
        return list(itertools.product(*self.states.values()))

    def _get_state_idx(self, RV):
        """Return ..."""

        # Return the column that corresponds to the position of 'RV'
        idx = np.array(self._get_index_tuples())
        return idx[:, self.variables.index(RV)]

    def _get_bool_idx(self, **kwargs):
        """Return ..."""

        # Only keep RVs that are in this factor's scope.
        states = {RV: state for RV, state in kwargs.items() if RV in self.scope}

        bools_per_RV = np.array([
            [s == state for s in self._get_state_idx(RV)]
            for RV, state in states.items()
        ])

        return bools_per_RV.all(axis=0)

    @property
    def display_name(self):
        names = list(self.states.keys())
        names = ','.join(names)

        return f'factor({names})'

    @property
    def cardinality(self):
        """Return the size of the dimensions of this Factor."""
        return self.values.shape

    @property
    def scope(self):
        """Return the scope of this factor."""
        return list(self.states.keys())

    # Alias
    variables = scope

    @property
    def vars(self):
        """Return the variables in this factor (i.e. the scope) as a *set*."""
        return set(self.scope)

    @property
    def width(self):
        """Return the width of this factor."""
        return len(self.scope)

    @property
    def flat(self):
        """Return the values as a flat list."""
        return self.values.reshape(-1)

    def reorder_scope(self, order, inplace=False):
        """Reorder the scope."""
        factor = self if inplace else Factor.copy(self)

        # rearranging the axes of 'factor' to match 'order'
        variables = factor.variables

        for axis in range(factor.values.ndim):
            exchange_index = variables.index(order[axis])

            variables[axis], variables[exchange_index] = (
                variables[exchange_index],
                variables[axis],
            )

            factor.values = factor.values.swapaxes(axis, exchange_index)

        factor.states = {key: factor.states[key] for key in order}

        if not inplace:
            return factor

    def align_index(self, other):
        """..."""
        indices = other._get_index_tuples()
        values = self[indices]
        return Factor(values, other.states)

    def extend_with(self, other, inplace=False):
        """Extend this factor with the variables & states of another."""
        factor = self if inplace else Factor.copy(self)

        # Note: the order of 'extra_vars' holds no importance ..
        extra_vars = set(other.variables) - set(factor.variables)

        if extra_vars:
            # Create as many new dimensions in the array as there are
            # additional variables.
            slice_ = [slice(None)] * len(factor.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            factor.values = factor.values[tuple(slice_)]

            factor.states.update(other.states)
            factor.name_to_number.update(other.name_to_number)
            factor.number_to_name.update(other.number_to_name)

        if not inplace:
            return factor

    def extend_and_reorder(self, factor, other):
        """Extend factors factor and other to be over the same scope and
        reorder their axes to match.
        """
        # Assuming 'other' is another Factor.
        other = Factor.copy(other)

        # modifying 'factor' (self) to add new variables
        factor.extend_with(other, inplace=True)

        # modifying 'other' to add new variables
        other.extend_with(factor, inplace=True)

        # rearranging the axes of 'other' to match 'factor'
        other.reorder_scope(factor.variables, inplace=True)

        # Since factor was modified in place, returning it is technically
        # unnecessary   .
        return factor, other

    def copy(self):
        """Return a copy of this Factor."""
        return Factor(self.values, self.states)

    def sum(self):
        """Sum all values of the factor."""
        return self.values.sum()

    def add(self, other, inplace=False):
        """A + B <=> A.add(B)"""
        factor = self if inplace else Factor.copy(self)

        if isinstance(other, (int, float)):
            factor.values += other

        else:
            # # Assuming 'other' is another Factor.
            factor, other = self.extend_and_reorder(factor, other)
            factor.values = factor.values + other.values

        if not inplace:
            return factor

    def mul(self, other, inplace=False):
        """A * B <=> A.mul(B)"""
        factor = self if inplace else Factor.copy(self)

        if isinstance(other, (int, float)):
            factor.values *= other

        else:
            # # Assuming 'other' is another Factor.
            factor, other = self.extend_and_reorder(factor, other)
            factor.values = factor.values * other.values

        if not inplace:
            return factor

    def div(self, other, inplace=False):
        """A / B <=> A.div(B)"""
        factor = self if inplace else Factor.copy(self)

        if isinstance(other, (int, float)):
            factor.values /= other

        else:
            # # Assuming 'other' is another Factor.
            factor, other = self.extend_and_reorder(factor, other)

            with warnings.catch_warnings(record=True) as w:
                # Cause all warnings to always be triggered.
                warnings.simplefilter("always")

                factor.values = factor.values / other.values
                factor.values[np.isnan(factor.values)] = 0

                # if len(w) > 0:
                #     print()
                #     print(w)
                #     print(factor.scope, factor.values)
                #     print(other.scope, other.values)

                # assert len(w) == 0

                # factor.values = values

        if not inplace:
            return factor

    def get_state_names(self, RV, idx):
        """Return the state for RV at idx."""
        return self.number_to_name[RV][idx]

    def del_state_names(self, RVs):
        """Deletes the state names for variables in RVs."""
        for RV in RVs:
            del self.states[RV]
            del self.name_to_number[RV]
            del self.number_to_name[RV]

    def get_state_index(self, RV, state):
        """Return the index for RV with state."""
        if isinstance(state, slice):
            return state

        if isinstance(state, (tuple, list)):
            return [self.name_to_number[RV][s] for s in state]

        return self.name_to_number[RV][state]

    def get(self, **kwargs):
        """..."""
        return self.flat[self._get_bool_idx(**kwargs)]

    def set(self, value, inplace=False, **kwargs):
        """Set a value to cells identified by **kwargs.

        Examples
        --------
        >>> factor = Factor([1, 1], {'A': ['a0', 'a1']})
        >>> print(factor)
        factor(A)
        A
        a0    1.0
        a1    1.0
        dtype: float64
        >>> factor.set(0, A='a0')
        >>> print(factor)
        factor(A)
        A
        a0    0.0
        a1    1.0
        dtype: float64
        """
        factor = self if inplace else Factor.copy(self)

        factor.values.reshape(-1)[factor._get_bool_idx(**kwargs)] = value

        if not inplace:
            return factor

    def set_complement(self, value, inplace=False, **kwargs):
        """Set a value to cells *not* identified by **kwargs.

        Examples
        --------
        >>> factor = Factor([1, 1], {'A': ['a0', 'a1']})
        >>> print(factor)
        factor(A)
        A
        a0    1.0
        a1    1.0
        dtype: float64
        >>> factor.set_complement(0, A='a0')
        >>> print(factor)
        factor(A)
        A
        a0    1.0
        a1    0.0
        dtype: float64
        """
        factor = self if inplace else Factor.copy(self)

        idx = np.invert(factor._get_bool_idx(**kwargs))
        factor.flat[idx] = value

        if not inplace:
            return factor

    def equals(self, other):
        """Return True iff two Factors have (roughly) the same values."""
        if not isinstance(other, Factor):
            return False

        if self.values.size != other.values.size:
            return False

        if set(self.scope) != set(other.scope):
            return False

        reordered = other.reorder_scope(self.scope).align_index(self)
        return np.allclose(self.values, reordered.values)

    def normalize(self, inplace=False):
        """Normalize the Factor so the sum of all values is 1."""
        factor = self if inplace else self.copy()

        factor.values = factor.values / factor.values.sum()

        if not inplace:
            return factor

    def sum_out(self, variables, simplify=False, inplace=False):
        """Sum-out (marginalize) a variable (or list of variables) from the
        factor.

        Args:
            variables (str, list): Name or list of names of variables to sum out.

        Returns:
            Factor: factor with the specified variable removed.
        """
        factor = self if inplace else Factor.copy(self)

        if isinstance(variables, (str, tuple)):
            variable_set = set([variables])
        else:
            variable_set = set(variables)

        if len(variable_set) == 0:
            # Nothing to sum out ...
            return factor

        scope = set(factor.variables)

        if not variable_set.issubset(scope):
            raise error.NotInScopeError(variable_set, scope)

        # Unstack the requested variables into columns and sum over them.
        var_indexes = [factor.variables.index(var) for var in variables]

        index_to_keep = set(range(len(factor.variables))) - set(var_indexes)
        factor.del_state_names(variables)

        factor.values = np.sum(factor.values, axis=tuple(var_indexes))

        if not inplace:
            return factor

    def project(self, Q, inplace=False):
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

        factor = self if inplace else Factor.copy(self)
        vars_to_sum_out = list(set(factor.scope) - Q)

        factor.sum_out(vars_to_sum_out, inplace=True)

        if not inplace:
            return factor

    # def zipped(self):
    #     """Return a dict with data, indexed by tuples."""
    #     return dict(zip(self._get_index_tuples(), self.values))

    @classmethod
    def from_series(cls, series):
        """Create a Factor from a (properly indexed) pandas.Series."""
        idx = series.index

        if isinstance(idx, pd.MultiIndex):
            states = {i.name: list(i) for i in idx.levels}
        else:
            states = {idx.name: list(idx)}

        return Factor(series.values, states)

    def as_series(self):
        """Return a pandas.Series."""
        idx = pd.MultiIndex.from_product(
            self.states.values(),
            names=self.states.keys()
        )

        return pd.Series(self.values.reshape(-1), index=idx)

    @classmethod
    def from_dict(cls, d):
        """Return a Factor initialized by its dict representation."""

        # Scope is guaranteed to be ordered in JSON
        scope = d['scope']

        states = {RV: d['states'][RV] for RV in scope}
        return Factor(d['data'], states)

    def as_dict(self):
        """Return a dict representation of this Factor."""

        return {
            'type': 'Factor',
            'scope': self.scope,
            'states': self.states,
            'data': self.values.tolist(),
        }

    @classmethod
    def from_data(cls, df, cols=None, states=None, complete_value=0):
        """Create a full Factor from data (using Maximum Likelihood Estimation).

        Determine the empirical distribution by counting the occurrences of
        combinations of variable states.

        Note that the this will *drop* any NAs in the data.

        Args:
            df (pandas.DataFrame): data
            cols (list): columns in the data frame to use. If `None`, all
                columns are used.
            variable_states (dict): list of allowed states for each random
                variable, indexed by name. If variable_states is None, `jpt`
                should be a pandas.Series with a proper Index/MultiIndex.
            complete_value (int): Base (count) value to use for combinations of
                variable states in the dataset.

        Return:
            Factor (unnormalized)
        """
        cols = cols if cols else list(df.columns)
        subset = df[cols]
        counts = subset.groupby(cols).size()

        if states is None:
            # We'll need to try to determine states from the jpt
            index = counts.index
            states = dict(zip(index.names, index.levels))

        # Create a factor containing *all* combinations set to `complete_value`.
        f2 = Factor(complete_value, states)

        # By summing the Factor with the Series all combinations not in the
        # data are set to `complete_value`.
        total = f2.as_series() + counts

        values = np.nan_to_num(total.values, nan=complete_value)
        return Factor(values, states=states)

