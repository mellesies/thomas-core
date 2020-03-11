# -*- coding: utf-8 -*-
"""BayesianNetwork"""
import sys, os
from datetime import datetime as dt

import itertools
from collections import OrderedDict

import networkx as nx
import networkx.algorithms.moral

import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

from .factor import Factor, mul
from .cpt import CPT
from .jpt import JPT

from .base import ProbabilisticModel
from .bag import Bag
from .junctiontree import JunctionTree, TreeNode

from . import error

import logging
log = logging.getLogger('thomas')



# ------------------------------------------------------------------------------
# BayesianNetwork
# ------------------------------------------------------------------------------
class BayesianNetwork(ProbabilisticModel):
    """A Bayesian Network (BN) consistst of Nodes and directed Edges.

    A BN is essentially a Directed Acyclic Graph (DAG) where each Node
    represents a Random Variable (RV) and is associated with a conditional
    probability theable (CPT). A CPT can only have a *single* conditioned
    variable; zero or more *conditioning*  variables are allowed. Conditioning
    variables are represented as the Node's parents.

    BNs can be used for inference. To do this efficiently, the BN first
    constructs a JunctionTree (or JoinTree).

    Because of the relation between the probaility distribution (expressed as a
    set of CPTs) and the graph structure, it is possible to instantiate a BN
    from a list of CPTs.
    """

    def __init__(self, name, nodes, edges):
        """Instantiate a new BayesianNetwork.

        Args:
            name (str): Name of the Bayesian Network.
            nodes (list): List of Nodes.
            edges (list): List of Edges.
        """
        self.name = name

        # dictionaries, indexed by nodes' random variables
        self.nodes = {}
        self.evidence = {}

        # Process the nodes and edges.
        if nodes:
            self.add_nodes(nodes)

            if edges:
                self.add_edges(edges)

        self.elimination_order = None

        # Cached junction tree
        self._jt = None

        # Widget ...
        self.__widget = None

    def __getitem__(self, RV):
        """x[name] <==> x.nodes[name]"""
        return self.nodes[RV]

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        s = f"<BayesianNetwork name='{self.name}'>\n"
        for RV in self.nodes:
            s += f"  <Node RV='{RV}' states={self.nodes[RV].states} />\n"

        s += '</BayesianNetwork>'

        return s

    # --- properties ---
    @property
    def edges(self):
        edges = []
        for n in self.nodes.values():
            for c in n._children:
                edges.append((n.RV, c.RV))

        return edges

    @property
    def junction_tree(self):
        """Return the junction tree for this network."""
        if self._jt is None:
            self._jt = JunctionTree(self)

        return self._jt

    # Aliases ...
    jt = junction_tree
    junctiontree = junction_tree
    jointree = junction_tree

    @property
    def vars(self):
        """Return the variables in this BN (i.e. the scope) as a *set*."""
        return set(self.nodes.keys())

    @property
    def scope(self):
        """Return the variables in this BN (i.e. the scope) as a *list*."""
        return list(self.nodes.keys())

    @property
    def states(self):
        """Return a dict of states, indexed by random variable."""
        return {RV: self.nodes[RV].states for RV in self.nodes}

    def setWidget(self, widget):
        """Associate this BayesianNetwork with a BayesianNetworkWidget."""
        self.__widget = widget

    # --- semi-private ---
    def _complete_case(self, case):
        """..."""
        missing = list(case[case.isna()].index)
        evidence = [e for e in case.index if e not in missing]

        try:
            probabilities = self.compute_posterior(
                missing,
                case.to_dict()
            )
        except error.InvalidStateError as e:
            print('WARNING - Could not complete case:', e)
            return np.NaN

        # Creata a dataframe by repeating the evicence multiple times
        imputated = pd.DataFrame([case[evidence]] * len(jpt), index=jpt.index)

        # The combinations of missing variables are in the index. Reset the
        # index to make them part of the dataframe.
        imputated = imputated.reset_index()

        # Add the computed probabilities as weights
        imputated.loc[:, 'weight'] = jpt.values

        return imputated

    def _complete_cases(self, incomplete_cases):
        # Create a dataframe that holds *all* unique cases. This way we can
        # complete these cases once and then just count their frequency.

        # drop_duplicates() will work fine with NaNs
        unique_cases = incomplete_cases.drop_duplicates().reset_index(drop=True)

        # apply() yields a Series where each entry holds a DataFrame.
        dfs = unique_cases.apply(self._complete_case, axis=1)
        dfs = dfs.dropna()

    # --- graph manipulation ---
    def add_nodes(self, nodes):
        """Add a Node to the network."""
        for node in nodes:
            self.nodes[node.RV] = node

        self._jt = None

    def add_edges(self, edges):
        """Recreate the edges using the nodes' CPTs."""
        for (parent_RV, child_RV) in edges:
            self.nodes[parent_RV].add_child(self.nodes[child_RV])

        self._jt = None

    def moralize_graph(self):
        """Return the moral graph for the DAG.

        A moral graph adds an edge between nodes that share a common child and
        then makes edges undirected.

        https://en.wikipedia.org/wiki/Moral_graph:
            > The name stems from the fact that, in a moral graph, two nodes
            > that have a common child are required to be married by sharing
            > an edge.
        """
        G = self.as_networkx()
        G_moral = nx.algorithms.moral.moral_graph(G)
        return list(G_moral.edges)

    # -- parameter estimation
    def EM_learning(self, data, reset_CPTs=True):
        """Perform parameter learning.
        Sources:
            * https://www.cse.ust.hk/bnbook/pdf/l07.h.pdf
            * https://www.youtube.com/watch?v=NDoHheP2ww4
        """
        pass

    def ML_estimation(self, df):
        """Perform Maximum Likelihood estimation of the BN parameters.

        *** Note: this will overwrite any CPTs already set. ***

        Args:
            df (pandas.Dataframe): dataset that contains columns with names
                corresponding to the variables in this BN's scope.
        """
        # Determine the empirical distribution by ..
        #   1. counting the occurrences of combinations for the variables in
        #      scope of this BN; and
        #   2. normalizing the result
        #
        # Note that the count opereation will also *drop* any NAs in the
        # dataframe.
        subset = df[self.scope]
        subset['count'] = 1

        counts = subset.groupby(self.scope).count()
        counts = counts['count']

        empirical = counts / counts.sum()

        # The empirical distribution may not contain all combinations of
        # variable states; this fixes that by setting all missing entries to 0.
        jpt = JPT.from_incomplete_jpt(empirical)

        for name, node in self.nodes.items():
            cpt = jpt.compute_dist(node.conditioned, node.conditioning)
            node.cpt = cpt

    # --- inference ---
    def get_node_elimination_order(self):
        """Return a naïve elimination ordering, based on nodes' degree."""
        if self.elimination_order is None:
            G = nx.Graph()
            G.add_edges_from(self.moralize_graph())

            degrees = list(G.degree)
            degrees.sort(key=lambda x: x[1])

            self.elimination_order = [d[0] for d in degrees]

        return self.elimination_order

    def compute_joint_with_jt(self, RVs):
        """Compute the joint distribution over multiple variables.

        Args:
            RVs (list or set): Set of random variables to compute joint over.

        Returns:
            Factor:  marginal distribution over the RVs in Q.
        """
        Q = set(RVs) if isinstance(RVs, list) else RVs

        jt = JunctionTree(self)
        jt.ensure_cluster(Q)
        node = jt.get_node_for_set(Q)

        joint = node.pull()

        if isinstance(RVs, list):
            joint = joint.reorder_scope(RVs)

        return joint

    def compute_marginals(self, qd=None, ev=None):
        """Compute the marginals of the query variables given the evidence.

        Args:
            qd (list): Random variables to query
            ev (dict): dict of states, indexed by RV to use as
                evidence.

        Returns:
            dict of marginals, indexed by RV
        """
        if ev is None:
            ev = {}

        # Reset the tree and apply evidence
        self.junction_tree.reset_evidence()
        self.junction_tree.set_evidence_hard(**ev)

        return self.junction_tree.get_marginals(qd)

    def compute_posterior(self, qd, qv, ed, ev, use_VE=False):
        """Compute the (posterior) probability of query given evidence.

        The query P(I,G=g1|D,L=l0) would imply:
            qd = ['I']
            qv = {'G': 'g1'}
            ed = ['D']
            ev = {'L': 'l0'}

        Args:
            qd (list): query distributions: RVs to query
            qv (dict): query values: RV-values to extract
            ed (list): evidence distributions: coniditioning RVs to include
            ev (dict): evidence values: values to set as evidence.

        Returns:
            CPT
        """
        # Evidence we can just set on the JT, but we'll need to compute the
        # joint over the other variables to answer the query.
        required_RVs = set(qd + list(qv.keys()) + ed)
        node = self.junction_tree.get_node_for_set(required_RVs)

        if node is None and use_VE == False:
            log.info('Cannot answer this query with the current junction tree.')
            use_VE = True

        if use_VE:
            log.debug('Using VE')
            return self.as_bag().compute_posterior(qd, qv, ed, ev)


        # Compute the answer to the query using the junction tree.
        log.debug(f'Found a node in the JT that contains {required_RVs}: {node.cluster}')
        self.junction_tree.reset_evidence()
        self.junction_tree.set_evidence_hard(**ev)

        # Process evidence distributions
        log.debug(f'  Projecting onto {required_RVs} without normalization')
        result = node.joint.project(set(required_RVs))
        result = result.normalize()

        if ed:
            result = result / result.project(set(ed))

        # RVs that are part of the query will need to be set as column names.
        query_vars = list(qv.keys()) + qd

        # If query values were specified we can extract them from the factor.
        if qv:
            result = result.extract_values(**qv)

        return CPT(result, conditioned_variables=query_vars)

    def reset_evidence(self, RVs=None, notify=True):
        """Reset evidence."""
        self.junction_tree.reset_evidence(RVs)

        if RVs:
            for RV in RVs:
                del self.evidence[RV]
        else:
            self.evidence = {}

        if self.__widget and notify:
            self.__widget.update()

    def set_evidence_likelihood(self, RV, **kwargs):
        """Set likelihood evidence on a variable."""
        self.junction_tree.set_evidence_likelihood(RV, **kwargs)
        self.evidence[RV] = kwargs

    def set_evidence_hard(self, RV, state, notify=True):
        """Set hard evidence on a variable.

        This corresponds to setting the likelihood of the provided state to 1
        and the likelihood of all other states to 0.
        """
        self.junction_tree.set_evidence_hard(**{RV: state})
        self.evidence[RV] = state

        if self.__widget and notify:
            self.__widget.update()


    def get_marginals(self, RVs=None):
        """Return the probabilities for a set off/all RVs given set evidence."""
        return self.junction_tree.get_marginals(RVs)

    # --- constructors ---
    @classmethod
    def from_CPTs(cls, name, CPTs):
        """Create a Bayesian Network from a list of CPTs."""
        nodes = {}
        edges = []

        for cpt in CPTs:
            RV = cpt.conditioned[0]
            node = DiscreteNetworkNode(RV)

            for parent_RV in cpt.conditioning:
                edges.append((parent_RV, RV))

            nodes[RV] = node

        bn = BayesianNetwork(name, nodes.values(), edges)

        for cpt in CPTs:
            RV = cpt.conditioned[0]
            bn[RV].cpt = cpt

        return bn

    # --- visualization ---
    def draw(self):
        """Draw the BN using networkx & matplotlib."""
        # nx.draw(self.as_networkx(), with_labels=True)

        nx_tree = self.as_networkx()
        pos = nx.spring_layout(nx_tree)

        nx.draw(
            nx_tree,
            pos,
            edge_color='black',
            font_color='white',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color='purple',
            alpha=1.0,
            with_labels=True,
        )

    def print_cpts(self):
        """Print all CPTs to stdout."""
        for name, node in self.nodes.items():
            print('-' * 60)
            print(node.cpt)
            print()

    # --- (de)serialization and conversion ---
    def as_networkx(self):
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        return G

    def as_bag(self):
        return Bag(
            name=self.name,
            factors=[n.cpt for n in self.nodes.values()]
        )

    def as_dict(self):
        """Return a dict representation of this Bayesian Network."""
        return {
            'type': 'BayesianNetwork',
            'name': self.name,
            'nodes': [n.as_dict() for n in self.nodes.values()],
            'edges': self.edges,
        }

    def as_json(self, pretty=False):
        """Return a JSON representation (str) of this Bayesian Network."""
        if pretty:
            indent = 4
        else:
            indent = None

        return json.dumps(self.as_dict(), indent=indent)

    def save(self, filename):
        with open(filename, 'w') as fp:
            fp.write(self.as_json(True))

    @classmethod
    def from_dict(self, d):
        """Return a Bayesian Network initialized by its dict representation."""
        name = d.get('name')
        nodes = [Node.from_dict(n) for n in d['nodes']]
        edges = d.get('edges')
        return BayesianNetwork(name, nodes, edges)

    @classmethod
    def from_json(cls, json_str):
        """Return a Bayesian Network initialized by its JSON representation."""
        d = json.loads(json_str)
        return cls.from_dict(d)

    @classmethod
    def open(cls, filename):
        with open(filename) as fp:
            data = fp.read()
            return cls.from_json(data)

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
        self.position = [0, 0]

        if cpt is not None:
            self.cpt = cpt

            if self.description == '':
                self.description = cpt.description
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
    def conditioned(self):
        """Return the conditioned variable(s)."""
        return self.cpt.conditioned

    @property
    def conditioning(self):
        """Return the conditioning variable(s)."""
        return self.cpt.conditioning

    @property
    def cpt(self):
        """Return the Node's CPT."""
        if self._cpt is None:
            # Create a new, uniform,CPT
            vs = {}

            for p in self.parents:
                vs[p.RV] = p.states

            vs[self.RV] = self.states

            self._cpt = CPT(1, variable_states=vs).normalize()

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
        if self.cpt.conditioning != [p.RV for p in self._parents]:
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
            'cpt': cpt,
            'position': self.position,
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
        node.position = d.get('position', (0,0))
        node.cpt = cpt

        return node



