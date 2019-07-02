# -*- coding: utf-8 -*-
"""Factor: the basis for all reasoning."""
import sys, os
from datetime import datetime as dt

import itertools
from collections import OrderedDict

import networkx as nx
import numpy as np
import pandas as pd
from pandas.core.dtypes.dtypes import CategoricalDtype
from functools import reduce

import json

import pybn
# from . import ProbabilisticModel
from ..factors.factor import Factor, mul
from ..factors.cpt import CPT
from ..factors.node import Node, DiscreteNetworkNode

from .bag import Bag
from .junctiontree import JunctionTree, TreeNode

from .. import error


# ------------------------------------------------------------------------------
# BayesianNetwork
# ------------------------------------------------------------------------------
class BayesianNetwork(object):
    """A Bayesian Network (BN) consistst of Nodes and directed Edges.

    A BN is essentially a Directed Acyclic Graph (DAG) where each Node 
    represents a Random Variable (RV) and is associated with a conditional 
    probability table (CPT). A CPT can only have a *single* conditioned 
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
        return {RV: self.nodes[RV].states for RV in self.nodes}

    # --- semi-private ---
    def _get_elimination_clusters_rec(self, edges=None, order=None):
        """Recursively compute the clusters for the elimination tree."""
        if edges is None:
            edges = self.moralize_graph()

        if order is None:
            order = self.get_node_elimination_order()

        if not len(order):
            return []

        # Reconstruct the graph
        G = nx.Graph()
        G.add_nodes_from(order)
        G.add_edges_from(edges)

        node = order.pop(0)

        # Make sure the neighbors from `node` are connected
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

    def _get_elimination_clusters(self, edges=None, order=None):
        """Compute the clusters for the elimination tree."""

        # Get the full set of clusters.
        clusters = self._get_elimination_clusters_rec(edges=edges, order=order)

        # Merge clusters that are contained in other clusters by iterating over
        # the full set and replacing the smaller with the larger clusters.
        merged = list(clusters)

        for _ in clusters:
            for outer_idx, outer_cluster in enumerate(merged):
                modified = False

                for inner_idx, inner_cluster in enumerate(merged):
                    if outer_idx == inner_idx:
                        continue

                    if inner_cluster.issubset(outer_cluster):
                        # print(inner_cluster, 'is a subset of', outer_cluster)
                        modified = True
                        break

                if modified:
                    # Break from the *middle* for loop.
                    break

            if modified:
                # We'll have to replace inner_cluster with outer_cluster to
                # maintain the running intersection property.
                merged[inner_idx] = merged[outer_idx]
                del merged[outer_idx]

            else:
                # Nothing was modified this iteration: it seems we're done.
                # Break from the *outermost* loop.
                break

        return merged
    
    def _compute_junction_tree(self):
        """Compute the junction tree for the current graph."""

        # First, create the JT itself.
        clusters = self._get_elimination_clusters()
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

                for node_j in remaining_nodes:
                    C_j = node_j.cluster

                    if intersection.issubset(C_j):
                        tree.add_edge(node_i, node_j)
                        break

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
        # for tree_node in nodes:
        #     for missing in (tree_node.cluster - tree_node.joint.vars):
        #         node = self.nodes[missing]
        #         states = {node.RV: node.states}
        #         trivial = Factor(1, variable_states=states)
        #         tree_node.add_factor(trivial)

        return tree

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

    # --- inference ---
    def moralize_graph(self):
        """Return the moral graph.

        FIXME: NetworkX has a method for this.    
        """
        edges = []

        for node in self.nodes.values():
            # Copy the list of parents
            parents = list(node._parents)

            edges += [(p.RV, node.RV) for p in parents]

            for outer_idx in range(len(parents)):
                p1 = parents[outer_idx]

                for inner_idx in range(outer_idx+1, len(parents)):
                    p2 = parents[inner_idx]
                    edges.append((p1.RV, p2.RV))

        return edges

    def get_node_elimination_order(self):
        """Return a naïve elimination ordering."""
        G = nx.Graph()
        G.add_edges_from(self.moralize_graph())

        degrees = list(G.degree)
        degrees.sort(key=lambda x: x[1])

        return [d[0] for d in degrees]

    def compute_posterior(self, query_dist, evidence_values=None):
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
        return self.junction_tree.get_probabilities(RVs)


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


