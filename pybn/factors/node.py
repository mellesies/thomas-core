# -*- coding: utf-8 -*-
"""Node: node in a Bayesian Network."""
import os
from datetime import datetime as dt

from collections import OrderedDict

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

from .factor import *
from .cpt import CPT

# ------------------------------------------------------------------------------
# Node
# ------------------------------------------------------------------------------
class Node(CPT):
    """Node in a Bayesian Network."""
    
    def __init__(self, name, cpt, description=''):
        """Initialize a new Node.

        Args:
            name (str): Name of the Node
            cpt (CPT, Factor, pandas.Series): CPT for this node. Can be one of
                CPT, Factor or pandas.Series. Factor or Series require an 
                appropriately set Index/MultiIndex.
        """
        if isinstance(cpt, CPT):
            # Do a sanity check and ensure that the CPTs has no more then a
            # single conditioned variable. This is only useful if cpt is an
            # actual CPT: for Factor/Series the last level in the index 
            # will be assumed to be the conditioned variable.
            error_msg = f'CPT should only have a single conditioned variable!'        
            assert len(cpt.conditioned) == 1, error_msg

            # Borrow the description from the CPT if not provided.
            if description == '':
                description = cpt.description

        # Call the super constructor
        super().__init__(cpt, description=description)

        # Initialize variables
        self.name = name

    @property
    def states(self):
        index = self._data_without_prefix.index

        if self.width == 1:
            return index.tolist()

        return index.levels[-1].tolist()

    @property
    def RV(self):
        """Return the name of the Random Variable for this Node."""
        return self.conditioned[0]

    def as_dict(self):
        """Return a dict representation of this Node."""
        d = super().as_dict()
        d.update({
            'type': 'Node',
            'name': self.name,
        })
        return d

    @classmethod
    def from_dict(cls, d):
        """Return a Node initialized by its dict representation."""
        name = d['name']
        cpt = super().from_dict(d)
        description = d['description']
        return Node(name, cpt, description)
    
