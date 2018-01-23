# -*- coding: utf-8 -*-
"""Example Bayesian networks."""

from collections import OrderedDict

import numpy as np
import pandas as pd

from . import Node, BayesianNetwork

def get_student_network():
    """Return the Student Bayesian Network."""

    # Set the conditional probabilities
    P = dict()
    I = np.matrix([0.7, 0.3])
    col_idx = pd.Index(['i0', 'i1'], name='I')
    P['I'] = pd.DataFrame(I, index=[''], columns=col_idx)

    S_I = np.matrix([[0.95, 0.05], 
                     [0.20, 0.80]])
    row_idx = pd.Index(['i0', 'i1'], name='I')
    col_idx = pd.Index(['s0', 's1'], name='S')
    P['S|I'] = pd.DataFrame(S_I, index=row_idx, columns=col_idx)

    D = np.matrix([0.6, 0.4])
    col_idx = pd.Index(['d0', 'd1'], name='D')
    P['D'] = pd.DataFrame(D, index=[''], columns=col_idx)

    G_DI = np.matrix([[0.30, 0.40, 0.30],
                      [0.05, 0.25, 0.70],
                      [0.90, 0.08, 0.02],
                      [0.50, 0.30, 0.20]])
    row_idx = pd.MultiIndex.from_product([['i0','i1'], ['d0','d1']], names=['I', 'D'])
    col_idx = pd.Index(['g1', 'g2', 'g3'], name='G')
    P['G|D,I'] = pd.DataFrame(G_DI, index=row_idx, columns=col_idx)

    L_G = np.matrix([[0.10, 0.90],
                     [0.40, 0.60],
                     [0.99, 0.01]])
    row_idx = pd.Index(['g1', 'g2', 'g3'], name='G')
    col_idx = pd.Index(['l0', 'l1'], name='L')
    P['L|G'] = pd.DataFrame(L_G, index=row_idx, columns=col_idx)

    # Define a node for each distribution
    nodes = {
        'I': Node('I', 'Intelligence', P['I']),
        'D': Node('D', 'Difficulty', P['D']),
        'G': Node('G', 'Grade', P['G|D,I']),
        'L': Node('L', 'Letter', P['L|G']),
        'S': Node('S', 'SAT', P['S|I']),
    }

    # Create edges/relations
    nodes['I'].add_child(nodes['G'])
    nodes['I'].add_child(nodes['S'])
    nodes['D'].add_child(nodes['G'])
    nodes['G'].add_child(nodes['L'])

    # Create and return the Bayesian Network
    return BayesianNetwork('Student', nodes)

