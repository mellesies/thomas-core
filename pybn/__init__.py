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


# ------------------------------------------------------------------------------
# Helper functions ... 
# ------------------------------------------------------------------------------
def as_list(item):
    if item is None:
        return []

    if isinstance(item, tuple):
        return list(item)
    
    if not isinstance(item, list):
        return [item]

    return item


# ------------------------------------------------------------------------------
# Factor
# ------------------------------------------------------------------------------
class Factor(object):
    """Factor."""

    def __init__(self, data, variable_states=None):
        """Initialize a new Factor.

        Args:
            data (list, pandas.Series): array of values.
            variable_states (dict): list of allowed states for each random 
                variable, indexed by name. If variable_states is None, `data` 
                should be a pandas.Series with a proper Index/MultiIndex.
        """
        if variable_states and len(variable_states) == 1:
            variable_states = list(variable_states.items())
            var_name, var_states = variable_states[0]
            idx = pd.Index(var_states, name=var_name)

        elif variable_states and len(variable_states) > 1:
            idx = pd.MultiIndex.from_product(
                variable_states.values(), 
                names=variable_states.keys()
            )

        elif (isinstance(data, pd.Series) 
                and isinstance(data.index, (pd.Index, pd.MultiIndex))):
            data = data.copy()
            idx = None

        else:
            msg =  'data should either be a pandas.Series *with* a proper'
            msg += ' index or variable_states should be provided.'
            print('Oh dear: ', type(data), type(data.index))
            print('variable_states: ', variable_states)
            raise Exception(msg)

        self._data = pd.Series(data, index=idx)
        # self._data.name = self.name

    def __repr__(self):
        """repr(f) <==> f.__repr__()"""
        return f'{self.name}\n{repr(self._data)}'

    def __mul__(self, other):
        """A * B <=> A.mul(B)"""
        return self.mul(other)

    def __truediv__(self, other):
        """A / B <=> A.div(B)"""
        return self.div(other)

    def __getitem__(self, *args, **kwargs):
        return self._data.__getitem__(*args, **kwargs)

    def sum(self):
        """Sum all values of the factor."""
        return self._data.sum()

    def mul(self, other, *args, **kwargs):
        """A * B <=> A.mul(B)"""
        if (isinstance(other, Factor) 
                and not self.overlaps_with(other)
                and self.name != other.name):
            return self.outer(other)

        return Factor(self._data.mul(other._data, *args, **kwargs))

    def div(self, other, *args, **kwargs):
        """A / B <=> A.mul(B)"""
        if isinstance(other, Factor):
            return Factor(self._data.div(other._data, *args, **kwargs))

        return Factor(self._data.div(other))

    def overlaps_with(self, other):
        own_scope = set(self.scope)
        other_scope = set(other.scope)
        result = len(own_scope.intersection(other_scope)) > 0

        # print('-' * 80)
        # print('own_scope:', own_scope)
        # print('other_scope:', other_scope)
        # print('result:', result)
        # print('-' * 80)

        return result

    @property
    def name(self):
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

        # Two ways to do this, not sure which one is faster ...
        unstacked = self._data.unstack(variable)
        summed = unstacked.sum(axis=1)

        # if isinstance(variable, str):
        #     variable = [variable]
        # variable = set(variable)
        # names = set(self.index.names)
        # summed = self.sum(level=list(names - variable))
        # summed.name = 'P({})'.format(','.join(summed.index.names))

        return Factor(summed)

    def stack(self, *args, **kwargs):
        """Proxy for pd.Series.stack()."""
        return self._data.stack(*args, **kwargs)

    def unstack(self, *args, **kwargs):
        """Proxy for pd.Series.unstack()."""
        return self._data.unstack(*args, **kwargs)

    def dot(self, other):
        if isinstance(other, Factor):
            return self._data.dot(other._data)

        # Hail mary ... ;)
        return self._data.dot(other)

    def outer(self, other):
        df = pd.DataFrame(
            np.outer(self._data, other._data),
            index=self._data.index, 
            columns=other._data.index
        )

        stacked = df.stack().squeeze()

        try:
            f = Factor(stacked)
        except:
            print('Could not create Factor from outer product?')
            print(type(stacked))
            print(stacked)
            raise

        return f

    def set_evidence(self, **kwargs):
        """Return a reduced factor."""

        # when called like zero(D='d1', E='e0'), we'll need to 
        # set rows that do not correspond to the evidence to zero.

        # Find the (subset of) evidence that's related to this factor's scope.
        levels = [l for l in kwargs.keys() if l in self.scope]
        data = self._data.copy()

        for l in levels:
            value = kwargs[l]
            idx = data.index.get_level_values(l) != value
            data[idx] = 0

        return Factor(data)

    def normalize(self):
        """Normalize the factor so the sum of all values is 1."""
        return self.__class__(self._data / self._data.sum())

    def sort_index(self, *args, **kwargs):
        return self._data.sort_index(*args, **kwargs)

    def as_series(self):
        return pd.Series(self._data)


# ------------------------------------------------------------------------------
# CPT: Conditional Probability Distribution
# ------------------------------------------------------------------------------
class CPT(Factor):
    """Conditional Probability Distribution.

    A CPT is essentially a Factor that knows which variables in its scope are
    the conditioning variables. We also *display* the CPT differently:
      - the random variable states make up the columns
      - the conditioning variable states make up the rows.
    """

    def __init__(self, data, conditioned_variables=None, 
        variable_states=None, description=''):
        """Initialize a new CPT.

        Args:
            data (list, pandas.Series): array of values.
            description (str): An optional description of the random variables' 
                meaning.
        """
        if isinstance(data, Factor):
            super().__init__(data._data)
        else:
            super().__init__(data, variable_states)

        self.description = description

        # Each name in the index corresponds to a random variable.
        variables = self.scope

        # We'll assume the last variable of the index is being conditioned if
        # not explicitly specified.
        if conditioned_variables is None:
            conditioned_variables = [variables[-1]]

        # The remaining variables must be the conditioning variables
        conditioning_variables = [i for i in variables if i not in conditioned_variables]

        self.conditioned = conditioned_variables
        self.conditioning = conditioning_variables

    @property
    def shortname(self):
        conditioned = ''.join(self.conditioned)
        conditioning = ''.join(self.conditioning)

        if conditioning:
            return f'{conditioned}_{conditioning}'
    
        return f'{conditioned}'
        
    @property
    def name(self):
        conditioned = ','.join(self.conditioned)
        conditioning = ','.join(self.conditioning)

        if conditioning:
            return f'P({conditioned}|{conditioning})'

        return f'P({conditioned})'

    def _repr_html_(self):
        if self.conditioning:
            html = self.unstack()._repr_html_()
        else:
            df = pd.DataFrame(self._data, columns=['']).transpose()
            html = df._repr_html_()

        return f"""
            <div>
                <div style='margin-top:6px'><b>{self.name}</b></div>
                {html}
            </div>
        """

    def reorder_scope(self, order=None):
        return CPT(
            super().reorder_scope(order),
            conditioned_variables=self.conditioned,
            description = self.description
        )

    def unstack(self, level=None, *args, **kwargs):
        if level is None:
            level = self.conditioned

        return super().unstack(level, *args, **kwargs)

    def dot(self, other):
        if isinstance(other, CPT):
            return self._data.dot(other.unstack())

        # Hail mary ... ;)
        return self._data.dot(other)

    def as_factor(self):
        """Return the Factor representation of this CPT."""
        return Factor(self._data)

# ------------------------------------------------------------------------------
# Bag: bag of factors
# ------------------------------------------------------------------------------
class Bag(object):
    """Bag  of factors."""

    def __init__(self, name='', factors=None):
        """Instantiate a new Bag."""
        self._factors = factors

    @property
    def scope(self):
        """Return the network scope."""
        network_scope = []

        for f in self._factors:
            network_scope += f.scope

        return set(network_scope)

    def find_elimination_ordering(self, Q):
        """Return a variable ordering for a set of factors."""
        return [v for v in self.scope if v not in Q]

    def prune(self, Q, e):
        """Dummy implementation to be overridden by subclasses."""
        return [f for f in self._factors]

    def eliminate(self, Q, e=None, debug=False):
        """Perform variable elimination."""        
        if e is None: e = {} 

        def mul(x1, x2): 
            """Helper function for functools.reduce()."""
            x1 = x1.reorder_scope()
            x2 = x2.reorder_scope()

            try:
                result = (x1 * x2)
            except: 
                print('-' * 80)
                print('could not multiply two factors')
                print(f'x1: {x1.name}: {x1.scope}')
                print(f'x2: {x2.name}: {x2.scope}')
                print('-' * 80)
                print()
                raise

            result = result.reorder_scope()
            return result

        # Initialize the list of factors to the pruned set. self.prune() should
        # be implemented by subclasses with more knowledge of the structure
        factors = self.prune(Q, e)

        # Apply the evidence to the pruned set.
        factors = [f.set_evidence(**e) for f in factors]

        # ordering will contain a list of variables *not* in Q, i.e. the 
        # remaining variables from the full distribution.
        ordering = self.find_elimination_ordering(Q)

        if debug:
            print('-' * 80)
            print(f'ordering: {ordering}')

        # Iterate over the variables in the ordering.
        for X in ordering:
            # Find factors that have the current variable 'X' in scope
            related_factors = [f for f in factors if X in f.scope]

            if debug:
                print('-' * 80)
                print(f'X: {X}')
                print('related_factors:', [f.name for f in related_factors])

            # Multiply all related factors with each other and sum out 'X'
            try:
                new_factor = reduce(mul, related_factors)

            except Exception as e:
                print()
                print('Could not reduce list of factors!?')
                # for f in related_factors:
                #     print(f'--- {f.name} ---')
                #     print(e)
                #     print(f)
                #     print()

                if (debug):
                    return related_factors

                # If we're not debugging, re-raise the Exception.
                raise e

            new_factor = new_factor.sum_out(X)

            # Replace the factors we have eliminated with the new factor. 
            factors = [f for f in factors if f not in related_factors]
            factors.append(new_factor)

        if debug:
            print('-' * 80)
            print('factors after main loop:', [f.name for f in factors])
            for f in factors:
                print(f)
                print()

        result = reduce(mul, factors)
        result = result.reorder_scope(Q)
        result.sort_index()

        return result


# ------------------------------------------------------------------------------
# Node
# ------------------------------------------------------------------------------
class Node(CPT):
    
    def __init__(self, cpt):
        """Initialize a new Node."""
        if isinstance(cpt, (Factor, pd.Series)):
            cpt = CPT(cpt)

        # First, do a sanity check and ensure that the CPTs has no more then
        # a single conditioned variable
        error_msg = f'CPT should only have a single conditioned variable!'        
        assert len(cpt.conditioned) == 1, error_msg

        # Call the super constructor
        super().__init__(cpt)

    @property
    def RV(self):
        """Return the name of the Random Variable for this Node."""
        return self.conditioned[0]
    
    
# ------------------------------------------------------------------------------
# BayesianNetwork
# ------------------------------------------------------------------------------
class BayesianNetwork(Bag):
    """A Bayesian Network (BN) consistst of Nodes and Edges.

    Conceptually a BN restricts/extends a Bag of factors by:
    - associating a node with each CPT; to make this possible it also ensures 
      that ...
    - every CPT only has a *single* conditioned variable
    - establishing relationships between the nodes by interpreting conditioning 
      variables to be the parents of the conditioned variable.
    """

    def __init__(self, name='', nodes=None):
        """Instantiate a new BayesianNetwork."""
        # Make sure nodes actually contains nodes
        nodes = [Node(n) for n in nodes]
        super().__init__(name, nodes)

        self.nodes = {n.RV: n for n in self._factors}
        self.edges = []

        for node in self._factors:
            for parent in node.conditioning:
                self.edges.append((parent, node.RV))

    def __getitem__(self, name):
        """x[name] <==> x.nodes[name]"""
        return self.nodes[name]

    def _parse_query_string(self, query_string):
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

    def compute_posterior(self, query_dist, query_values, evidence_dist, 
        evidence_values):
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
        # Get a list of *all* variables to query
        query_vars = query_dist + list(query_values.keys())
        evidence_vars = evidence_dist + list(evidence_values.keys())

        # First, compute the joint over the query variables and the evidence.
        # result = self.eliminate(query_vars + evidence_dist, evidence_values)
        result = self.eliminate(query_vars + evidence_vars)
        result = result.normalize()

        # print('result: ', result)

        # At this point, result's scope is over all query and evidence variables
        # If we're computing an entire conditional distribution ...
        # if evidence_dist:
        if evidence_vars:
            try:
                result = result / result.sum_out(query_vars)
            except:
                print('-' * 80)
                print(f'trying to sum out {query_vars}')
                print(result)
                print('-' * 80)
                raise

        # print('result: ', result)


        # If query values were specified we can extract them from the factor.
        if query_values:
            levels = list(query_values.keys())
            values = list(query_values.values())

            if result.width == 1:
                result = result[values[0]]

            elif result.width > 1:
                # print(f'values: {values}', f'levels: {levels}')
                # print(result)
                indices = []

                for level, value in query_values.items():
                    # print('-' * 80)
                    # print(level, value)
                    # print('-' * 80)
                    idx = result._data.index.get_level_values(level) == value
                    indices.append(list(idx))

                # result = Factor(result._data.xs(values, level=levels))

                zipped = list(zip(*indices))
                idx = [all(x) for x in zipped]
                result = Factor(result._data[idx])
                # print('indices: ', indices)
                # print('zipped:', zipped)
                # print('idx: ', idx)
                # print('result: ', result)
                # print('-' * 80)


        if evidence_values:
            indices = []

            for level, value in evidence_values.items():
                idx = result._data.index.get_level_values(level) == value
                indices.append(list(idx))
                # print('-' * 80)
                # print(level, value)
                # print('-' * 80)

            zipped = list(zip(*indices))
            idx = [all(x) for x in zipped]
            # print('indices: ', indices)
            # print('zipped:', zipped)
            # print('idx: ', idx)
            # print('-' * 80)
            result = Factor(result._data[idx])

        if isinstance(result, Factor):
            order = list(evidence_vars) + list(query_vars)
            result = result.reorder_scope(order)
            result.sort_index()
            return CPT(result, conditioned_variables=query_vars)

        return result

    def P(self, query_string):
        """Return the probability as queried by query_string.

        P('I,G=g1|D,L=l0') is equivalent to calling compute_posterior with:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}
        """
        qd, qv, gd, gv = self._parse_query_string(query_string)
        return self.compute_posterior(qd, qv, gd, gv)

    def prune(self, Q, e):
        """Prune the graph."""
        edges = list(self.edges)
        factors = list(self._factors)

        should_continue_pruning = False

        while should_continue_pruning:
            pass

        return super().prune(Q, e)
