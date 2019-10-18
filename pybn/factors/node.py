# -*- coding: utf-8 -*-
"""Node: node in a Bayesian Network."""
import sys
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
class Node(object):
    """Base class for discrete and continuous nodes in a Bayesian Network.

    In Hugin, discrete nodes can only have other discrete nodes as parents.
    Continous nodes can have either continuous or discrete nodes as parents.

    BayesiaLab does allow discrete nodes to have continous nodes as parents by
    associating discrete states with the continous value.
    """

    def __init__(self, RV, name=None, description=''):
        """Initialize a new Node.

        Args:
            RV (str): Name of the (conditioned) random variable
            name (str): Name of the Node.
            description (str): Name of the Node
        """
        self.RV = RV
        self.name = name or RV
        self.description = description

        # A node needs to know its parents in order to determine the shape of
        # its CPT. This should be a list of Nodes.
        self._parents = []

        # For purposes of message passing, a node also needs to know its
        # children.
        self._children = []

    @property
    def parents(self):
        return self._parents

    def add_parent(self, parent, add_child=True):
        """Add a parent to the Node.

        If succesful, the Node's distribution's parameters (ContinousNode) or
        CPT (DiscreteNode) should be reset.

        Args:
            parent (Node): parent to add.
            add_child (bool): iff true, this node is also added as a child to
                the parent.

        Return:
            True iff the parent was added.
        """
        if parent not in self._parents:
            self._parents.append(parent)

            if add_child:
                parent.add_child(self, add_parent=False)

            return True

        return False

    def add_child(self, child, add_parent=True):
        """Add a child to the Node.

        Args:
            child (Node): child to add.
            add_child (bool): iff true, this node is also added as a parent to
                the child.

        Return:
            True iff the child was added.
        """
        if child not in self._children:
            self._children.append(child)

            if add_parent:
                child.add_parent(self, add_child=False)

            return True

        return False

    def remove_parent(self, parent, remove_child=True):
        """Remove a parent from the Node.

        If succesful, the Node's distribution's parameters (ContinousNode) or
        CPT (DiscreteNode) should be reset.

        Return:
            True iff the parent was removed.
        """
        if parent in self._parents:
            self._parents.remove(parent)

            if remove_child:
                parent._children.remove(self)

            return True

        return False

    def validate(self):
        """Validate the probability parameters for this Node."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, d):
        """Return a Node (subclass) initialized by its dict representation."""
        clsname = d['type']

        if clsname == cls.__name__:
            raise Exception('Cannot instantiate abstract class "Node"')

        clstype = getattr(sys.modules[__name__], clsname)
        return clstype.from_dict(d)

# ------------------------------------------------------------------------------
# DiscreteNetworkNode
# ------------------------------------------------------------------------------
class DiscreteNetworkNode(Node):
    """Node in a Bayesian Network with discrete values."""

    def __init__(self, RV, name=None, states=None, description='', cpt=None):
        """Initialize a new discrete Node.

        A Node represents a random variable (RV) in a Bayesian Network. For
        this purpose, it keeps track of a conditional probability distribution
        (CPT).

        Args:
            name (str): Name of the Node. Should correspond to the name of a
                conditioned variable in the CPT.
            states (list): List of states (strings)
            description (str): Name of the Node
        """
        super().__init__(RV, name, description)

        self.states = states or []

        if cpt is not None:
            self.cpt = cpt
        else:
            self._cpt = None

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        components = [f"DiscreteNetworkNode('{self.RV}'"]

        if self.name:
            components.append(f"name='{self.name}'")

        if self.states:
            states = ', '.join([f"'{s}'" for s in self.states])
            components.append(f"states=[{states}]")

        if self.description:
            components.append(f"description='{self.description}'")

        return ', '.join(components) + ')'

    @property
    def parents(self):
        if self._cpt:
            parents = dict([(p.RV, p) for p in self._parents])
            sort_order = list(self._cpt._data.index.names[:-1])

            return [parents[p] for p in sort_order]

        return self._parents

    @property
    def cpt(self):
        """Return the Node's CPT."""
        return self._cpt

    @cpt.setter
    def cpt(self, cpt):
        """
        Set the Node's CPT.

        This method should only be called *after* the node's parents are known!

        Args:
            cpt (CPT, Factor, pandas.Series): CPT for this node. Can be one of
                CPT, Factor or pandas.Series. Factor or Series require an
                appropriately set Index/MultiIndex.
        """

        # Do a sanity check and ensure that the CPTs has no more then a
        # single conditioned variable. This is only useful if cpt is an
        # actual CPT: for Factor/Series the last level in the index
        # will be assumed to be the conditioned variable.
        if not isinstance(cpt, CPT):
            e = "Argument should be a CPT"
            raise Exception(e)

        elif len(cpt.conditioned) != 1:
            e = "CPT should only have a single conditioned variable"
            raise Exception(e)

        elif cpt.conditioned[0] != self.RV:
            e = "Conditioned variable in CPT should correspond to Node's RV"
            raise Exception(e)

        # elif self.states and self.states != cpt.variable_states[self.RV]:
        #     e = "States in CPT should match the Node's states.\n"
        #     e += f" -> Node: {self.states}\n"
        #     e += f" -> CPT: {cpt.variable_states[self.RV]}\n"
        #     raise Exception(e)

        if not self.states:
            self.states = cpt.variable_states[self.RV]

        # Looking good :-)
        self._cpt = cpt

    @property
    def vars(self):
        """Return the variables in this node (i.e. the scope) as a set."""
        if self._cpt:
            return self._cpt.vars

        return []

    def reset(self):
        """Create a default CPT.

        Throws an Exception if states is not set on this Node or one of its
        parents.
        """
        states = {}

        # Iterate over the parents (all DiscreteNodes) and the node itself to
        # create a dict of states. In Python ≥ 3.6 these dicts are ordered!
        for p in (self._parents + [self, ]):
            if not p.states:
                msg = 'Cannot reset the values of Node (with a parent) without'
                msg += ' states!'
                raise Exception(msg)

            states[p.name] = p.states

        # Assume a uniform distribution
        self.cpt = CPT(1, variable_states=states).normalize()

    def add_parent(self, parent, **kwargs):
        """Add a parent to the Node.

        Discrete nodes can only have other discrete nodes as parents. If
        succesful, the Node's CPT will be reset.

        Return:
            True iff the parent was added.
        """
        e = "Parent of a DiscreteNetworkNode should be a DiscreteNetworkNode."
        e += f" Not a {type(parent)}"
        assert isinstance(parent, DiscreteNetworkNode), e

        if super().add_parent(parent, **kwargs):
            return True

        return False

    def remove_parent(self, parent):
        """Remove a parent from the Node.

        If succesful, the Node's CPT will be reset.

        Return:
            True iff the parent was removed.
        """
        if super().remove_parent(parent):
            self.reset()
            return True

        return False

    def validate(self):
        """Validate the probability parameters for this Node."""
        if cpt.conditioning != [p.RV for p in self._parents]:
            e  = "Conditioning variables in CPT should correspond to Node's"
            e += " parents. Order is important!"
            raise Exception(e)


    # --- (de)serialization ---
    def as_dict(self):
        """Return a dict representation of this Node."""
        cpt = self.cpt.as_dict() if self.cpt else None

        d = {
            'type': 'DiscreteNetworkNode',
            'RV': self.RV,
            'name': self.name,
            'states': self.states,
            'description': self.description,
            'cpt': cpt
        }

        return d

    @classmethod
    def from_dict(cls, d):
        """Return a DiscreteNetworkNode initialized by its dict representation."""
        cpt = CPT.from_dict(d['cpt'])

        node = DiscreteNetworkNode(
            RV=d['RV'],
            name=d['name'],
            states=d['states'],
            description=d['description']
        )

        node.cpt = cpt

        return node
