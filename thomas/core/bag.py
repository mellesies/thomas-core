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

from .base import ProbabilisticModel, remove_none_values_from_dict
from .factor import Factor, mul
from .cpt import CPT

from . import error

import logging
log = logging.getLogger('thomas')

# ------------------------------------------------------------------------------
# Bag
# ------------------------------------------------------------------------------
class Bag(ProbabilisticModel):
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

    def __len__(self):
        """len(f) == f.__len__()"""
        return len(self._factors)

    @staticmethod
    def _scope(factors):
        """Return the scope of a set of factors.

        This comprises the set of (unique) variables covered by the factors.
        """
        # One liner flattens the list of lists
        return set([item for factor in factors for item in factor.scope])

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
        scope = self._scope(factors)
        return [v for v in scope if v not in Q]

    def eliminate(self, Q, evidence=None):
        """Perform variable elimination."""
        if evidence is None:
            evidence = {}

        # Initialize the list of factors and apply the evidence
        factors = list(self._factors)

        # factors = [f.keep_values(**evidence) for f in factors]
        factors = [f.set_complement(0, **evidence) for f in factors]

        # ordering will contain a list of variables *not* in Q, i.e. the
        # remaining variables from the full distribution.
        ordering = self.find_elimination_ordering(Q, factors)

        # Iterate over the variables in the ordering.
        for X in ordering:
            # Find factors that have the current variable 'X' in scope
            related_factors = [f for f in factors if X in f.scope]

            # Multiply all related factors with each other and sum out 'X'
            new_factor = reduce(mul, related_factors)
            new_factor = new_factor.sum_out(X)

            # Replace the factors we have eliminated with the new factor.
            factors = [f for f in factors if f not in related_factors]
            factors.append(new_factor)

        result = reduce(mul, factors)

        if isinstance(result, Factor):
            try:
                result = result.reorder_scope(Q)
                # result.sort_index()
            except Exception as e:
                log.error(f'Could not reorder scope: {e}')
                log.error(result.scope)
                pass

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
        ev = remove_none_values_from_dict(ev)

        # Get a list of *all* variables to query
        query_vars = list(qv.keys()) + qd
        evidence_vars = list(ev.keys()) + ed

        # First, compute the joint over the query variables and the evidence.
        result = self.eliminate(query_vars + ed, ev)
        result = result.normalize()

        # At this point, result's scope is over all query and evidence variables
        # If we're computing an entire conditional distribution ...
        if evidence_vars:
            result = result / result.sum_out(query_vars)

        # If query values were specified we can extract them from the factor.
        if qv:
            result = result.get(**qv)

        if isinstance(result, Factor):
            return CPT(
                result.values,
                states=result.states,
                conditioned=query_vars
            )

        elif isinstance(result, np.ndarray) and len(result) == 1:
            result = result[0]

        return result

    # --- (de)serialization and conversion ---
    def as_dict(self):
        """Return a dict representation of this Bag."""
        return {
            'type': 'Bag',
            'name': self.name,
            'factors': [f.as_dict() for f in self._factors]
        }

