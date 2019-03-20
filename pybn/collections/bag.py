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
from ..factors.factor import Factor
from ..factors.cpt import CPT
from ..factors.node import Node

from .. import error as e

# ------------------------------------------------------------------------------
# Bag: bag of factors
# ------------------------------------------------------------------------------
class Bag(object):
    """Bag  of factors."""

    def __init__(self, name='', factors=None):
        """Instantiate a new Bag."""
        self.name = name
        self._factors = factors

    def __repr__(self):
        return f"<Bag: '{self.name}'>"

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
        e = pybn.add_prefix_to_dict(e)

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

        def mul_debug(x1, x2):
            print('-' * 80)
            print("I got chiiiiils, they're multiplying")
            print('-' * 80)
            print(f'x1: {x1.scope}')
            print(x1)
            print('-' * 80)
            print(f'x2: {x2.scope}')
            print(x2)
            print('-' * 80)
            print()
            return mul(x1,x2)


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
            print(f'Q: {Q}')
            print(f'ordering: {ordering}')

        # Iterate over the variables in the ordering.
        for X in ordering:
            # Find factors that have the current variable 'X' in scope
            related_factors = [f for f in factors if X in f.scope]

            if debug:
                print('-' * 80)
                print('LOOP')
                print(f'X: {X}')
                print('related_factors:')
                for f in related_factors:
                    print(f'  - {f.scope}')

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
            print('FINAL')
            print('factors after main loop:')
            for f in factors:
                print(f'  - {f.scope}')

            print('-' * 80)
            # return factors

        if debug:
            result = reduce(mul_debug, factors)
            print('result.scope:', result.scope)
            print('-' * 80)
        else:
            result = reduce(mul, factors)

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

    def as_dict(self):
        """Return a dict representation of this Bag."""
        return {
            'type': 'Bag',
            'name': self.name,
            'factors': [f.as_dict() for f in self._factors]
        }

