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

from .. import error as e

# ------------------------------------------------------------------------------
# Node
# ------------------------------------------------------------------------------
class Node(CPT):
    """Node in a Bayesian Network."""
    
    def __init__(self, cpt, name='', description=''):
        """Initialize a new Node."""
        if isinstance(cpt, (Factor, pd.Series)) and not isinstance(cpt, CPT):
            cpt = CPT(cpt, description=description)

        # First, do a sanity check and ensure that the CPTs has no more then
        # a single conditioned variable
        error_msg = f'CPT should only have a single conditioned variable!'        
        assert len(cpt.conditioned) == 1, error_msg

        # Call the super constructor
        super().__init__(cpt, description=cpt.description or description)

        # Initialize variables
        self.name = name

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
        cpt = super().from_dict(d)
        return Node(cpt)
    
