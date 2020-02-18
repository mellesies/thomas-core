# -*- coding: utf-8 -*-
"""Factor: the basis for all reasoning."""
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

import pybn
# from . import ProbabilisticModel
from ..factor.factor import Factor, mul
from ..factor.cpt import CPT
from ..factor.node import Node, DiscreteNetworkNode

from .bag import Bag
from .junctiontree import JunctionTree, TreeNode

from .. import error

import logging
log = logging.getLogger('pybn')



def get_fill_in_edges(edges, order, fill_in=None):
    """Recursively compute the clusters for the elimination tree."""
    # Make sure we're not modifying the method argument.
    if fill_in is None:
        fill_in = list()

    order = list(order)

    if order == []:
        return fill_in

    # Reconstruct the graph
    G = nx.Graph()
    G.add_nodes_from(order)
    G.add_edges_from(edges)

    node = order.pop(0)

    # Make sure the neighbors from `node` are connected by adding fill-in
    # edges.
    neighbors = list(G.neighbors(node))

    if len(neighbors) > 1:
        for outer_idx in range(len(neighbors)):
            n1 = neighbors[outer_idx]

            for inner_idx in range(outer_idx+1, len(neighbors)):
                n2 = neighbors[inner_idx]
                G.add_edge(n1, n2)
                fill_in.append((n1, n2))

    G.remove_node(node)
    cluster = set([node, ] + neighbors)

    return get_fill_in_edges(G.edges, order, fill_in)

def get_cluster_sequence(edges, order):
    """Recursively compute the clusters for the elimination tree."""
    # Make sure we're not modifying the method argument.
    order = list(order)

    if not len(order):
        return []

    # Reconstruct the graph
    G = nx.Graph()
    G.add_nodes_from(order)
    G.add_edges_from(edges)

    node = order.pop(0)

    # Make sure the neighbors from `node` are connected by adding fill-in
    # edges.
    neighbors = list(G.neighbors(node))

    if len(neighbors) > 1:
        for outer_idx in range(len(neighbors)):
            n1 = neighbors[outer_idx]

            for inner_idx in range(outer_idx+1, len(neighbors)):
                n2 = neighbors[inner_idx]
                G.add_edge(n1, n2)

    G.remove_node(node)
    cluster = set([node, ] + neighbors)

    return [cluster, ] + get_cluster_sequence(G.edges, order)

def running_intersection(clusters):
    for idx_i, C_i in enumerate(clusters[:-1]):
        right_union = set.union(*clusters[idx_i+1:])
        intersection = C_i.intersection(right_union)

        print(f'idx_i:        {idx_i}')
        print(f'cluster:      {C_i}')
        print(f'right_union:  {right_union}')
        print(f'intersection: {intersection}')

        for c in clusters[idx_i+1:]:
            if intersection.issubset(c):
                print(f'contained in: {c}')
                break

        print()

def merge_clusters(clusters):
    clusters = list(clusters)
    should_continue = True

    while should_continue:
        should_continue = False

        for idx_i in range(len(clusters)):
            modified = False
            C_i = clusters[idx_i]

            for idx_j in range(idx_i+1, len(clusters)):
                C_j = clusters[idx_j]

                if C_j.issubset(C_i):
                    clusters[idx_j] = C_i
                    clusters.pop(idx_i)

                    modified = True
                    should_continue = len(clusters) > 1
                    break

            if modified:
                break

    return clusters


# ------------------------------------------------------------------------------
# BayesianNetwork
# ------------------------------------------------------------------------------
class BayesianNetwork(object):
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

        # dictionary, indexed by nodes' random variables
        self.nodes = {}

        # Process the nodes and edges.
        if nodes:
            self.add_nodes(nodes)

            if edges:
                self.add_edges(edges)

        self.elimination_order = None

        # Cached junction tree
        self._jt = None

    def __getitem__(self, RV):
        """x[name] <==> x.nodes[name]"""
        return self.nodes[RV]

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        s = f"<BayesianNetwork name='{self.name}'>\n"
        for RV in self.nodes:
            s += f"  <Node RV='{RV}' />\n"

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
            self._jt = self._compute_junction_tree()

        return self._jt

    @property
    def states(self):
        """Return a dict of states, indexed by random variable."""
        return {RV: self.nodes[RV].states for RV in self.nodes}

    # --- semi-private ---
    def _get_elimination_clusters(self, edges=None, order=None):
        """Compute the clusters for the elimination tree."""

        # Get the full set of clusters.
        clusters = self._get_elimination_clusters_rec(edges=edges, order=order)

        # Merge clusters that are contained in other clusters by iterating over
        # the full set and replacing the smaller with the larger clusters.
        merged = list(clusters)
        clusters = list(clusters)

        # We should merge later clusters into earlier clusters.
        # The reversion is undone just before the function returns.
        clusters.reverse()

        should_continue = len(clusters) > 1
        while should_continue:
            should_continue = False

            for idx_i in range(len(clusters)):
                modified = False
                C_i = clusters[idx_i]

                for idx_j in range(idx_i+1, len(clusters)):
                    C_j = clusters[idx_j]

                    if C_i.issubset(C_j):
                        clusters[idx_i] = C_j
                        clusters.pop(idx_j)

                        modified = True
                        should_continue = len(clusters) > 1
                        break

                if modified:
                    break

        # Undo the earlier reversion.
        clusters.reverse()
        return clusters

    def _get_elimination_clusters_rec(self, edges=None, order=None):
        """Recursively compute the clusters for the elimination tree."""
        if edges is None:
            edges = self.moralize_graph()

        if order is None:
            order = self.get_node_elimination_order()

        # Make sure we're not modifying the method argument.
        order = list(order)
        # print(f'using elimination_order: {order}')

        if not len(order):
            return []

        # Reconstruct the graph
        G = nx.Graph()
        G.add_nodes_from(order)
        G.add_edges_from(edges)

        node = order.pop(0)

        # Make sure the neighbors from `node` are connected by adding fill-in
        # edges.
        neighbors = list(G.neighbors(node))

        if len(neighbors) > 1:
            for outer_idx in range(len(neighbors)):
                n1 = neighbors[outer_idx]

                for inner_idx in range(outer_idx+1, len(neighbors)):
                    n2 = neighbors[inner_idx]
                    G.add_edge(n1, n2)

        G.remove_node(node)
        cluster = set([node, ] + neighbors)

        return [cluster, ] + self._get_elimination_clusters_rec(G.edges, order)

    def _compute_junction_tree(self, order=None):
        """Compute the junction tree for the current graph."""

        # First, create the JT itself.
        clusters = self._get_elimination_clusters(order=order)
        tree = JunctionTree(clusters)
        nodes = list(tree.nodes.values())

        for idx, node_i in reversed(list(enumerate(nodes))):
            C_i = node_i.cluster

            # We'll compute union(C_i+1 , C_i+2, .... C_n)
            # 'remaining' will hold clusters C_i+1 , C_i+2, .... C_n
            remaining_nodes = nodes[idx+1:]

            if remaining_nodes:
                remaining_clusters = [n.cluster for n in remaining_nodes]
                intersection = C_i.intersection(set.union(*remaining_clusters))

                found = False

                for node_j in remaining_nodes:
                    C_j = node_j.cluster

                    if intersection.issubset(C_j):
                        tree.add_edge(node_i, node_j)
                        found = True
                        break

                if not found:
                    print('*** WARNING ***')
                    print('Could not add node_i: ', node_i)
                    print('C_i:', C_i)
                    print('remaining_clusters:')
                    for r in remaining_clusters:
                        print('  ', r)
                    print()

        # Iterate over all nodes in the BN to assign each BN node/CPT to the
        # first JT node that contains the BN node's RV. Also, assign an evidence
        # indicator for that variable to that JT node.
        # Any other JT node that contains the BN node's RV, should be given the
        # trivial factor (all 1s).
        indicators = {}

        # Iterate over the nodes in the BN
        for RV, node in self.nodes.items():
            first = True

            # Iterate over the JT nodes/clusters
            for tree_node in nodes:
                if node.vars.issubset(tree_node.cluster):
                    tree_node.add_factor(node.cpt)
                    tree.set_node_for_RV(RV, tree_node)

                    states = {RV: node.states}
                    indicator = Factor(1, variable_states=states)
                    tree.add_indicator(indicator, tree_node)
                    break

        # Iterate over the JT nodes/clusters to make sure each cluster has
        # the correct factors assigned.
        # FIXME: I think this is superfluous for minimal JTs?
        for tree_node in nodes:
            try:
                for missing in (tree_node.cluster - tree_node.vars):
                    node = self.nodes[missing]
                    states = {node.RV: node.states}
                    trivial = Factor(1, variable_states=states)
                    tree_node.add_factor(trivial)
            except:
                print('*** WARNING ***')
                print('tree_node.cluster:', tree_node.cluster)
                print('tree_node.vars:', tree_node.vars)

        return tree

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

    # --- testing only
    def FE1(self, Q, order=None):
        if order is None:
            order = self.nodes.keys()

        # Don't modify the original list
        order = list(order)

        # Transform the network into a list of CPTs
        S = [self.nodes[n].cpt for n in order]

        # Find a node that has Q in it and assign it as root
        for cpt in S:
            if Q in cpt.scope:
                root = cpt
                # print(f'Using node {root.display_name} as root')
                break

        # Move the root node to the end/beginning.
        # This saves some checking later.
        S.remove(root)
        S.append(root)
        S.reverse()

        while len(S) > 1:
            f_i = S.pop()
            # print(f'Removed factor {f_i.display_name} from S')

            vars_in_S = set()
            for cpt in S:
                vars_in_S = vars_in_S | cpt.vars

            V = f_i.vars - vars_in_S

            if len(V):
                # print(f'Summing out {V}')
                f_i = f_i.sum_out(V)
            else:
                # print('No variables to sum out ...')
                pass

            f_j = S[-1]

            # print(f'Multiplying {f_i.display_name} into {f_j.display_name}')

            S[-1] = f_j * f_i
            # print()

        return S[0].project(Q)

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
        """Return the moral graph."""
        G = self.as_networkx()
        G_moral = nx.algorithms.moral.moral_graph(G)
        return list(G_moral.edges)

    # --- inference ---
    def get_node_elimination_order(self):
        """Return a naïve elimination ordering."""
        if self.elimination_order is None:
            G = nx.Graph()
            G.add_edges_from(self.moralize_graph())

            degrees = list(G.degree)
            degrees.sort(key=lambda x: x[1])

            self.elimination_order = [d[0] for d in degrees]

        return self.elimination_order

    def compute_posterior(self, query_dist=None, evidence_values=None):
        """Compute the probability of the query variables given the evidence.

        :param tuple query_dist: Random variable to query
        :param dict evidence_values: Conditioned on values
        :return: dict, indexed by RV
        """
        if evidence_values is None:
            evidence_values = {}

        # Reset the tree and apply evidence
        self.junction_tree.reset_evidence()

        for RV, state in evidence_values.items():
            self.junction_tree.set_evidence_hard(RV, state)

        return self.junction_tree.get_probabilities(query_dist)

    def P(self, query_string, always_use_VE=False):
        """Return the probability as queried by query_string.

        P('I,G=g1|D,L=l0') is equivalent to calling compute_posterior with:
            query_dist = ('I',)
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}
        """
        qd, qv, ed, ev = self._parse_query_string(query_string)
        self.junction_tree.reset_evidence()

        if not always_use_VE:
            required_RVs = set(qd + list(qv.keys() )+ ed)

            for node in self.junction_tree.nodes.values():

                if required_RVs.issubset(node.cluster):
                    log.debug(f'Found a node: {node.cluster}')
                    query_vars = list(qv.keys()) + qd

                    for RV, value in ev.items():
                        self.junction_tree.set_evidence_hard(RV, value)

                    if ed:
                        # The cluster may contained variables we're not
                        # interested in
                        superfluous = node.cluster - required_RVs
                        result = node.joint

                        if superfluous:
                            log.debug(f'Summing out {superfluous}')
                            result = result.sum_out(superfluous)

                        # Divide by the evidence distribution
                        result = result / node.joint.project(set(ed))

                    else:
                        log.debug(f'Projecting onto {qd}')
                        result = node.joint.project(set(qd))

                    result = result.normalize()


                    # FIXME: this code is duplicated from bag.py
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

                    cpt = pybn.CPT(result, conditioned_variables=query_vars)

                    log.debug('Used JT')
                    return cpt

        # If answering the query requires modification of the JT, fall
        # back to variable elimination.
        log.debug('Used VE')
        return self.as_bag().P(query_string)

    def reset_evidence(self, RV=None):
        """Reset evidence for one or more RVs."""
        self.junction_tree.reset_evidence(RV)

    def set_evidence_likelihood(self, RV, **kwargs):
        """Set likelihood evidence on a variable."""
        self.junction_tree.set_evidence_likelihood(RV, **kwargs)

    def set_evidence_hard(self, RV, state):
        """Set hard evidence on a variable.

        This corresponds to setting the likelihood of the provided state to 1
        and the likelihood of all other states to 0.
        """
        self.junction_tree.set_evidence_hard(RV, state)

    def get_probability(self, RV):
        return self.junction_tree.get_probability(RV)

    def get_probabilities(self, RVs=None):
        """Return the probabilities for a set off/all RVs given set evidence."""
        return self.junction_tree.get_probabilities(RVs)

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

    # --- (de)serialization and conversion ---
    def as_networkx(self):
        G = nx.DiGraph()
        G.add_edges_from(self.edges)
        return G

    def as_bag(self):
        return pybn.Bag(
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


