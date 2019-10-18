# -*- coding: utf-8 -*-
"""JunctionTree"""
import networkx as nx
from functools import reduce

from . import ProbabilisticModel

from ..util import sep
from ..factors.factor import mul, Factor
# ------------------------------------------------------------------------------
# JunctionTree
# ------------------------------------------------------------------------------
class JunctionTree(object):
    """JunctionTree for a BayesianNetwork.


    The tree consists of TreeNodes and TreeEdges.
    """

    def __init__(self, clusters=None):
        """Initialize a new JunctionTree."""
        if clusters is None:
            clusters = []

        self.clusters = clusters
        self.nodes = {}      # TreeNode, indexed by cluster.label
        self.edges = []
        self.indicators = {} # evidence indicators; indexed by RV
        self._RVs = {}       # TreeNode, indexed by RV and

        for c in self.clusters:
            self.add_node(c)

    def get_node_for_RV(self, RV):
        """A[x] <==> A.__getitem__(x)"""
        return self._RVs[RV]

    def set_node_for_RV(self, RV, node):
        """A[x] = y <==> A.__setitem__(x, y)"""
        self._RVs[RV] = node

    def get_probability(self, RV):
        return self.get_node_for_RV(RV).project(RV)

    def get_probabilities(self, RVs=None):
        """Return the probabilities for a set off/all RVs given set evidence."""
        if RVs is None:
            RVs = self._RVs

        return {RV: self.get_node_for_RV(RV).project(RV) for RV in RVs}

    def add_node(self, cluster, factors=None, evidence=False):
        """Add a node to the tree."""

        # FIXME: I don't think `evidence` is ever set to True in the
        #        current implementation.
        if evidence:
            node = EvidenceNode(cluster, factors)
        else:
            node = TreeNode(cluster, factors)

        self.nodes[node.label] = node
        return node

    def add_edge(self, node1, node2):
        """Add an edge between two nodes."""
        if isinstance(node1, str):
            node1 = self.nodes[node1]

        if isinstance(node2, str):
            node2 = self.nodes[node2]

        self.edges.append(TreeEdge(node1, node2))
        # Adding an edge requires us to recompute the separators.

    def add_indicator(self, factor, node):
        """Add an indicator for a random variable to a node."""
        if isinstance(node, str):
            node = self.nodes[node]

        RV = list(factor.variable_states.keys()).pop()
        self.indicators[RV] = factor
        node.indicators.append(factor)

    def reset_evidence(self, RVs=None):
        """Reset evidence for one or more RVs."""
        if RVs is None:
            RVs = self.indicators

        elif isinstance(RVs, str):
            RVs = [RVs]

        for RV in RVs:
            indicator = self.indicators[RV]
            indicator._data[:] = 1.0

        self.invalidate_caches()

    def set_evidence_likelihood(self, RV, **kwargs):
        """Set likelihood evidence on a variable."""
        indicator = self.indicators[RV]

        # FIXME: it's not pretty to access Factor._data like this!
        data = indicator._data

        for state, value in kwargs.items():
            data[state] = value

        self.invalidate_caches()

    def set_evidence_hard(self, RV, state):
        """Set hard evidence on a variable.

        This corresponds to setting the likelihood of the provided state to 1
        and the likelihood of all other states to 0.
        """
        indicator = self.indicators[RV]

        if state not in indicator.index.get_level_values(RV):
            state = state.replace(f'{RV}.', '')
            raise e.InvalidStateError(RV, state, self)

        # FIXME: it's not pretty to access Factor._data like this!
        data = indicator._data
        idx = data.index.get_level_values(RV) != state
        data[idx] = 0.0

        idx = data.index.get_level_values(RV) == state
        data[idx] = 1.0

        self.invalidate_caches()

    def invalidate_caches(self):
        """Invalidate the nodes' caches."""
        for n in self.nodes.values():
            n.invalidate_cache()

    def run(self):
        for n in self.nodes.values():
            n.pull()

    def compute_posterior(self, query_dist, query_values, evidence_dist,
        evidence_values, **kwargs):
        """Compute the probability of the query variables given the evidence.

        The query P(I,G=g1|D,L=l0) would imply:
            query_dist = ['I']
            query_values = {'G': 'g1'}
            evidence_dist = ('D',)
            evidence_values = {'L': 'l0'}

        :param tuple query_dist: Random variable to query
        :param dict query_values: Random variable values to query
        :param tuple evidence_dist: Conditioned on evidence
        :param dict evidence_values: Conditioned on values
        :return: pandas.Series (possibly with MultiIndex)
        """
        Q = set(query_dist)
        for n in self.nodes.values():
            if Q in n.cluster:
                return n

    def as_networkx(self):
        """Return the JunctionTree as a networkx.Graph() instance."""
        G = nx.Graph()

        for e in self.edges:
            G.add_edge(e._left, e._right, label=','.join(e.separator))

        return G

    def draw(self):
        """Draw the JunctionTree using networkx & matplotlib."""
        nx_tree = self.as_networkx()
        pos = nx.spring_layout(nx_tree)

        nx.draw(
            nx_tree,
            pos,
            edge_color='black',
            width=1,
            linewidths=1,
            node_size=1500,
            node_color='pink',
            alpha=1.0,
            labels={node:node.label for node in nx_tree.nodes}
        )

        nx.draw_networkx_edge_labels(
            nx_tree,
            pos,
            edge_labels={key:value['label'] for key,value in nx_tree.edges.items()},
            font_color='red'
        )

# ------------------------------------------------------------------------------
# TreeEdge
# ------------------------------------------------------------------------------
class TreeEdge(object):
    """Edge in an elimination/junction tree."""

    def __init__(self, left, right):
        """Initialize a new TreeEdge.

        Note that it's not actually left or right.
        """
        self._left = left
        self._right = right

        self._separator = None

        left.add_neighbor(self)
        right.add_neighbor(self)

    def __repr__(self):
        """repr(x) <==> x.__repr__()"""
        return f'Edge: ({repr(self._left)} - {repr(self._right)})'

    def __getitem__(self, key):
        """a[key] <==> a.__getitem__(key) <==> a.get_neighbor(key)"""
        return self.get_neighbor(key)

    @property
    def separator(self):
        """Return/compute the separator on this edge."""
        if self._separator is None:
            self.recompute_separator()

        return self._separator

    def get_neighbor(self, node):
        """Return the neighbor for `node`."""
        if node == self._left:
            return self._right

        if node == self._right:
            return self._left

        raise Exception('Supplied node is not connected to this edge!?')

    def recompute_separator(self):
        """(re)compute the separator for this Edge."""
        left_downstream = self._left.get_all_downstream_nodes(self)
        right_downstream = self._right.get_all_downstream_nodes(self)

        left_cluster = set.union(*[n.cluster for n in left_downstream])
        right_cluster = set.union(*[n.cluster for n in right_downstream])

        self._separator = set.intersection(left_cluster, right_cluster)

# ------------------------------------------------------------------------------
# TreeNode
# ------------------------------------------------------------------------------
class TreeNode(object):
    """Node in an elimination/junction tree."""

    def __init__(self, cluster, factors=None):
        """Initialize a new node."""
        self.cluster = cluster
        self.indicators = []

        if factors:      # list: Factor
            self._factors = factors
        else:
            self._factors = []

        self._edges = [] # list: TreeEdge

        self._joint = None   # cached joint distribution over self._factors

        # The cache is indexed by upstream node.
        self.invalidate_cache() # sets self._cache = {}

    def __repr__(self):
        """x.__repr__() <==> repr(x)"""
        # return self.label
        return f"TreeNode({self.cluster})"

    @property
    def label(self):
        """Return the Node's label."""
        return ','.join(self.cluster)

    @property
    def factors(self):
        return self._factors + self.indicators

    @property
    def vars(self):
        v = [f.vars for f in self.factors]
        if v:
            return set.union(*v)
        return set()

    @property
    def joint(self):
        """Compute the joint of the Node's factors."""
        if self._joint is None:
            factors = self._factors + self.indicators

            # Per documentation for reduce: "If initializer is not given and
            # sequence contains only one item, the first item is returned."
            try:
                self._joint = reduce(mul, factors)
            except:
                print('*** ERROR ***')
                print('Error while trying to compute the joint distribution')
                print(f'Node: {self.cluster}')
                print(f'Factors:', factors)
                raise

        return self._joint

    def add_neighbor(self, edge):
        if edge not in self._edges:
            self._edges.append(edge)

    def add_factor(self, factor):
        """Add a factor to this Node."""
        self._factors.append(factor)
        self._joint = None

    def invalidate_cache(self):
        """Invalidate the cache.

        This comprises the message cache and the joint distribution for the
        cluster with indicators.
        """
        self._cache = {}
        self._joint = None

    def get_downstream_edges(self, upstream=None):
        return [e for e in self._edges if e is not upstream]

    def get_all_downstream_nodes(self, upstream=None):
        edges = self.get_downstream_edges(upstream)

        if upstream is None:
            downstream = {}

            # edges is a list: TreeEdge
            for e in edges:
                downstream[e] = e.get_neighbor(self).get_all_downstream_nodes(e)

            return downstream

        downstream = []

        for edge in edges:
            node = edge.get_neighbor(self)
            downstream.extend(node.get_all_downstream_nodes(edge))

        return [self, ] + downstream

    def pull(self, upstream=None):
        """Trigger pulling of messages towards this node."""
        downstream_edges = self.get_downstream_edges(upstream)
        result = self.joint

        if downstream_edges:
            downstream_results = []

            for e in downstream_edges:
                if e not in self._cache:
                    n = e.get_neighbor(self)
                    self._cache[e] = n.pull(e)

                downstream_results.append(self._cache[e])

            result = reduce(mul, downstream_results + [result])

        if upstream:
            return result.project(upstream.separator)

        return result

    def project(self, RV, normalize=True):
        """Trigger a pull and project the result onto RV.

        RV should be contained in the Node's cluster.
        """
        result = self.pull().project(RV)

        if not normalize:
            return result

        return result.normalize()

# ------------------------------------------------------------------------------
# EvidenceNode
# ------------------------------------------------------------------------------
class EvidenceNode(TreeNode):
    """Node that can be used to set evidence or compute the prior."""

    def __init__(self, cluster, factors=None):
        """Initialize a new node."""
        msg = "EvidenceNode cluster can only contain a single variable."
        assert len(cluster) == 1, msg
        assert factors is None or len(factors) == 1, msg

        super().__init__(cluster, factors)

    @property
    def factor(self):
        return self._factors[0]

    def add_factor(self, factor):
        if len(factors) > 0:
            raise Exception('EvidenceNode can only contain a single variable.')

        super().add_factor(factors)

    def add_neighbor(self, edge):
        if edge not in self._edges:
            if len(self._edges) > 1:
                raise Exception('EvidenceNode can only contain a single variable.')

            super().add_neighbor(edge)

