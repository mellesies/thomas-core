"""Methods for learning the structure of a bayesian network from data"""
import numpy as np
import pandas as pd

from collections import namedtuple
from itertools import combinations
from sklearn.metrics import mutual_info_score


from .factor import Factor
from .cpt import CPT

def greedy_network_learning(df, degree_network=2):
    """Learns the network using a greedy approach based on the mutual
    information between variables

    Args:
        df (pandas.Dataframe): dataset that contains columns with names
            corresponding to the variables in this BN's scope.
        degree_network (int): maximum number of parents each node can
            have.
    """

    # init network
    network = []
    nodes = set(df.columns)
    nodes_selected = set()

    # define structure of NodeParentPair candidates
    NodeParentPair = namedtuple('NodeParentPair', ['node', 'parents'])

    # select random node as starting point
    root = np.random.choice(tuple(nodes))
    network.append(NodeParentPair(node=root, parents=None))
    nodes_selected.add(root)

    # select each remaining node iteratively that have the highest
    # mutual information with nodes that are already in the network
    for i in range(len(nodes_selected), len(nodes)):
        nodes_remaining = nodes - nodes_selected
        n_parents = min(degree_network, len(nodes_selected))

        node_parent_pairs = [
            NodeParentPair(n, tuple(p)) for n in nodes_remaining
            for p in combinations(nodes_selected, n_parents)
        ]

        # compute the mutual information for each note_parent pair candidate
        scores = _compute_scores(df, node_parent_pairs)

        # add best scoring candidate to the network
        sampled_pair = node_parent_pairs[np.argmax(scores)]
        nodes_selected.add(sampled_pair.node)
        network.append(sampled_pair)
    return network

def compute_cpts_network(df, network):
    """Computes the conditional probability distribution of each node
    in the Bayesian network"""
    P = dict()
    for idx, pair in enumerate(network):
        if pair.parents is None:
            cpt = CPT.from_factor(Factor.from_data(df, cols=[pair.node])).normalize()
            # cpt = CPT(marginal_distribution, conditioned=[pair.node]).normalize()
        else:
            # todo: there should be a from_data at CPT
            cpt = CPT.from_factor(Factor.from_data(df, cols=[*pair.parents, pair.node])).normalize()
            # cpt = CPT(joint_distribution, conditioned=[pair.node]).normalize()

        # add conditional distribution to collection
        P[pair.node] = cpt
    return P


# def compute_cpts_network(df, network):
#     """Computes the conditional probability distribution of each node
#     in the Bayesian network"""
#     P = dict()
#     for idx, pair in enumerate(network):
#         if pair.parents is None:
#             marginal_counts = df.groupby([pair.node]).size()
#             marginal_distribution = marginal_counts / marginal_counts.sum()
#             cpt = CPT.from_factor(Factor.from_series(marginal_distribution)).normalize()
#             # cpt = CPT(marginal_distribution, conditioned=[pair.node]).normalize()
#         else:
#             npp_columns = [*pair.parents, pair.node]
#             df_npp = df[npp_columns]
#             # todo: can we do this with thomas?
#             # compute joint distribution for NodeParentPair
#             index = [df_npp[c] for c in df_npp.columns[:-1]]
#             column = df_npp[df_npp[npp_columns].columns[-1]]
#             contingency_table = pd.crosstab(index, column, dropna=False).stack()
#             joint_distribution = contingency_table / contingency_table.sum()
#
#             if isinstance(joint_distribution, pd.Series):
#                 joint_distribution = joint_distribution.to_frame()
#             # todo: there should be a from_data at CPT
#             cpt = CPT.from_factor(Factor.from_data(joint_distribution)).normalize()
#             # cpt = CPT(joint_distribution, conditioned=[pair.node]).normalize()
#
#         # add conditional distribution to collection
#         P[pair.node] = cpt
#     return P

def _compute_scores(df, node_parent_pairs):
    """Computes mutual information for all NodeParentPair candidates"""
    scores = np.empty(len(node_parent_pairs))
    for idx, pair in enumerate(node_parent_pairs):
        scores[idx] = _compute_mutual_information(df, pair)
    return scores

def _compute_mutual_information(df, pair):
    node_values = df[pair.node].values
    if len(pair.parents) == 1:
        parent_values = df[pair.parents[0]].values
    else:
        # combine multiple parent columns into one string column
        parent_values = df.loc[:, pair.parents].astype(str).apply(lambda x: '-'.join(x.values), axis=1).values
    return mutual_info_score(node_values, parent_values)



# def _compute_mutual_information(df, pair):
#     """Computes mutual information between existing child and parent nodes"""
#     p_node = _positive_probability(df, columns=[pair.node])
#     p_parents =_positive_probability(df, columns=list(pair.parents))
#     p_nodeparents = _positive_probability(df, columns=[*pair.parents, pair.node])
#
#     #fixme: outer function should be multiplication
#     mi = np.sum(p_nodeparents.values * np.log((p_nodeparents / (outer(p_node, p_parents)).values)))
#     return mi
#
# def _positive_probability(df, columns):
#     counts = df.groupby(columns).size()
#     return counts / counts.sum()
#
# def outer(x1, x2):
#     """Return the outer product."""
#     df = pd.DataFrame(
#         np.outer(x1, x2),
#         index=x1._data.index,
#         columns=x2._data.index
#     )
#
#     if isinstance(df.columns, pd.MultiIndex):
#         levels = df.columns.levels
#         stacked = df.stack(list(range(len(levels)))).squeeze()
#     else:
#         stacked = df.stack().squeeze()
#
#     f = Factor.from_series(stacked)
#     return f


# fixme: depreciated as Factor now has 0 counts which will result in -inf when taking the logartihm
# def _compute_mutual_information(df, pair):
#     """Computes mutual information between existing child and parent nodes"""
#     p_node = Factor.from_data(df[[pair.node]]).normalize()
#     p_parents = Factor.from_data(df[list(pair.parents)]).normalize()
#     p_nodeparents = Factor.from_data(df[[*pair.parents, pair.node]]).normalize()
#
#     mi = np.sum(p_nodeparents.values * np.log((p_nodeparents / (p_node * p_parents)).values))
#     return mi