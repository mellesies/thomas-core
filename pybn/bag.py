# -*- coding: utf-8 -*-
"""Bag: collection of Factors."""
import os
from datetime import datetime as dt

from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

import pybn
from .factor import Factor, mul
from .cpt import CPT

from . import error

# ------------------------------------------------------------------------------
# Bag
# ------------------------------------------------------------------------------
class Bag(pybn.ProbabilisticModel):
    """Bag of factors."""

    def __init__(self, name='', factors=None):
        """Instantiate a new Bag."""
        msg = f'factors should be a list, not a {type(factors)}'
        assert isinstance(factors, list), msg

        self.name = name
        self._factors = factors

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        return f"<Bag: '{self.name}'>"

    def _scope(self, factors):
        """Return the scope of a set of factors.

        This comprises the set of (unique) variables covered by the factors.
        """
        scope = []

        for f in factors:
            scope += f.scope

        return set(scope)

    # --- properties ---
    @property
    def scope(self):
        """Return the network scope."""
        return self._scope(self._factors)

    # --- inference ---
    def find_elimination_ordering(self, Q, factors):
        """Return a variable ordering for a set of factors.

        The result will only contain variables *not* in Q.
        """
        # filtered = [f for f in factors if not f.overlaps_with(Q)]
        # filtered.sort(key=lambda x: x.width)

        scope = self._scope(factors)
        return [v for v in scope if v not in Q]

    def eliminate(self, Q, e=None, debug=False):
        """Perform variable elimination."""
        if e is None: e = {}

        # Initialize the list of factors to the pruned set. self.prune() should
        # be implemented by subclasses with more knowledge of the structure
        factors = list(self._factors)

        # Apply the evidence to the pruned set.
        try:
            factors = [f.keep_values(**e) for f in factors]
        except error.InvalidStateError as e:
            # Actually, don't deal with this here ...
            raise

        # ordering will contain a list of variables *not* in Q, i.e. the
        # remaining variables from the full distribution.
        ordering = self.find_elimination_ordering(Q, factors)

        if debug:
            print('-' * 80)
            print(f'Q: {Q}')
            print(f'ordering: {ordering}')
            print('all factors:', [f.display_name for f in factors])

        # Iterate over the variables in the ordering.
        for X in ordering:
            # Find factors that have the current variable 'X' in scope
            related_factors = [f for f in factors if X in f.scope]

            if debug:
                print('-' * 80)
                print('LOOP')
                print(f'X: {X}')
                print('factors:')
                for f in factors:
                    print('-' * 10)
                    print(f)
                print('-' * 40)
                print('related_factors:')
                for f in related_factors:
                    print('-' * 10)
                    print(f)

            # Multiply all related factors with each other and sum out 'X'
            try:
                new_factor = reduce(mul, related_factors)

            except Exception as e:
                print()
                print('Could not reduce list of factors!?')

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
            print('FINAL')
            print('factors after main loop:')
            for f in factors:
                print(f'  - {f.scope}')

            print('-' * 80)

        result = reduce(mul, factors)

        if debug:
            print('result.scope:', result.scope)
            print('-' * 80)

        try:
            result = result.reorder_scope(Q)
        except:
            print('exception while reordering scope')
            print('Q:', Q)
            print('result.scope:', result.scope)
            print('result:', result)
            raise

        result.sort_index()

        return result

    def compute_posterior(self, qd, qv, ed, ev):
        """Compute the probability of the query variables given the evidence.

        The query P(I,G=g1|D,L=l0) would imply:
            qd = ['I']
            qv = {'G': 'g1'}
            ed = ('D',)
            ev = {'L': 'l0'}

        Args:
            qd (list): query distributions: RVs to query
            qv (dict): query values: RV-values to extract
            ed (list): evidence distributions: coniditioning RVs to include
            ev (dict): evidence values: values to set as evidence.

        Returns:
            CPT
        """
        ev = pybn.remove_none_values_from_dict(ev)

        # Get a list of *all* variables to query
        query_vars = list(qv.keys()) + qd
        evidence_vars = list(ev.keys()) + ed

        # First, compute the joint over the query variables and the evidence.
        result = self.eliminate(query_vars + ed, ev)
        result = result.normalize()

        # At this point, result's scope is over all query and evidence variables
        # If we're computing an entire conditional distribution ...
        if evidence_vars:
            try:
                result = result / result.sum_out(query_vars)
            except:
                print('-' * 80)
                print(f'trying to sum out {query_vars}')
                print(result)
                print('-' * 80)
                raise

        # If query values were specified we can extract them from the factor.
        if qv:
            levels = list(qv.keys())
            values = list(qv.values())

            if result.width == 1:
                result = result[values[0]]

            elif result.width > 1:
                indices = []

                for level, value in qv.items():
                    idx = result._data.index.get_level_values(level) == value
                    indices.append(list(idx))

                zipped = list(zip(*indices))
                idx = [all(x) for x in zipped]
                result = Factor(result._data[idx])

        if isinstance(result, Factor):
            result.sort_index()
            return CPT(result, conditioned_variables=query_vars)

        return result

    # --- (de)serialization and conversion ---
    def as_dict(self):
        """Return a dict representation of this Bag."""
        return {
            'type': 'Bag',
            'name': self.name,
            'factors': [f.as_dict() for f in self._factors]
        }

