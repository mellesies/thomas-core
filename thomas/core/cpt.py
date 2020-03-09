# -*- coding: utf-8 -*-
"""CPT: Conditional Probability Table."""
import os
from datetime import datetime as dt

from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

from .factor import *
from . import error as e


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

    def __init__(self, data, conditioned_variables=None, variable_states=None, description=''):
        """Initialize a new CPT.

        Args:
            data (list, pandas.Series, Factor): array of values.
            conditioned_variables (list): list of conditioned variables
            variable_states (dict): list of allowed states for each random
                variable, indexed by name. If variable_states is None, `data`
                should be a pandas.Series (or Factor) with a proper
                Index/MultiIndex.
            description (str): An optional description of the random variables'
                meaning.
        """
        if isinstance(data, Factor):
            super().__init__(data._data, data._variable_states)
        else:
            super().__init__(data, variable_states)

        # Each name in the index corresponds to a random variable. We'll assume
        # the last variable of the index is being conditioned if not explicitly
        # specified.
        if conditioned_variables is None:
            conditioned = [self.scope[-1]]
        else:
            conditioned = conditioned_variables

        # The remaining variables must be the conditioning variables
        conditioning = [i for i in self.scope if i not in conditioned]

        # Make sure the conditioned variable appears rightmost in the index.
        if self.width > 1:
            order = conditioning + conditioned
            self._data = self._data.reorder_levels(order)

        # Sort the index
        self._data = self._data.sort_index()

        # Set remaining attributes
        self.conditioned = conditioned
        self.conditioning = conditioning
        self.description = description

    @classmethod
    def _short_query_str(cls, sep1, sep2, conditioned, conditioning):
        """Return a short query string."""
        conditioned = sep1.join(conditioned)
        conditioning = sep1.join(conditioning)

        if conditioning:
            return f'{conditioned}{sep2}{conditioning}'

        return f'{conditioned}'

    def short_query_str(self, sep1=',', sep2='|'):
        """Return a short version of the query string."""
        return self._short_query_str(
            sep1,
            sep2,
            self.conditioned,
            self.conditioning
        )

    @property
    def display_name(self):
        """Return a short version of the query string."""
        return f'P({self.short_query_str()})'

    def _repr_html_(self):
        """Return an HTML representation of this CPT."""
        data = self._data

        if self.conditioning:
            html = data.unstack(self.conditioned)._repr_html_()
        else:
            df = pd.DataFrame(data, columns=['']).transpose()
            html = df._repr_html_()

        return f"""
            <div>
                <div style="margin-top:6px">
                    <span><b>{self.display_name}</b></span>
                    <span style="font-style: italic;">{self.description}</span>
                    {html}
                </div>
            </div>
        """

    def normalize(self):
        """Normalize the factor so the sum of all values is 1."""
        if len(self.conditioning) >= 1:
            return self.__class__(self._data / self._data.unstack().sum(axis=1))

        return self.__class__(self._data / self._data.sum())

    def unstack(self, level=None, *args, **kwargs):
        if level is None:
            level = self.conditioned

        return super().unstack(level, *args, **kwargs)

    # melle: This wasn't used in any of the real code, only in the notebooks
    #        as example.
    # def dot(self, other):
    #     if isinstance(other, CPT):
    #         return self._data.dot(other.unstack())
    #
    #     # Hail mary ... ;)
    #     return self._data.dot(other)

    def as_factor(self):
        """Return the Factor representation of this CPT."""
        return Factor(self._data)

    def as_dict(self):
        """Return a dict representation of this CPT."""
        d = super().as_dict()
        d.update({
            'type': 'CPT',
            'description': self.description,
            'conditioned': self.conditioned,
            'conditioning': self.conditioning
        })

        return d

    @classmethod
    def from_dict(cls, d):
        """Return a CPT initialized by its dict representation."""
        factor = super().from_dict(d)

        return CPT(
            factor,
            conditioned_variables=d.get('conditioned'),
            description=d.get('description')
        )


