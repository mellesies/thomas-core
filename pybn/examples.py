# -*- coding: utf-8 -*-
"""Example Bayesian networks."""

import numpy as np
import pandas as pd

from pybn import Factor, CPT, Node, BayesianNetwork

def subset(full_dict, keys):
    return {k: full_dict[k] for k in keys}

def get_student_network():
    """Return the Student Bayesian Network."""

    # Set the conditional probabilities
    P = dict()
    states = {
        'I': ['i0', 'i1'],
        'S': ['s0', 's1'],
        'D': ['d0', 'd1'],
        'G': ['g1', 'g2','g3'],
        'L': ['l0', 'l1'],
    }

    P['I'] = CPT(
        [0.7, 0.3], 
        variable_states=subset(states, ['I'])
    )

    P['S|I'] = CPT(
        [0.95, 0.05, 
         0.20, 0.80], 
        variable_states=subset(states, ['I', 'S'])
    )

    P['D'] = CPT(
        [0.6, 0.4], 
        variable_states=subset(states, ['D'])
    )

    P['G|DI'] = CPT(
        [0.30, 0.40, 0.30, 
         0.05, 0.25, 0.70, 
         0.90, 0.08, 0.02, 
         0.50, 0.30, 0.20],
        variable_states=subset(states, ['I', 'D', 'G'])
    )

    P['L|G'] = CPT(
        [0.10, 0.90,
         0.40, 0.60,
         0.99, 0.01],
        variable_states=subset(states, ['G', 'L'])
    )

    return BayesianNetwork('Student', P.values())


def get_sprinkler_network():

    states = {
        'A': ['a1', 'a0'],
        'B': ['b1', 'b0'],
        'C': ['c1', 'c0'],
        'D': ['d1', 'd0'],
        'E': ['e1', 'e0'],
    }

    # P(A)
    fA = Factor(
        [0.6, 0.4], 
        subset(states, ['A'])
    )

    # P(B|A)
    fB_A = Factor(
        [0.2, 0.8, 0.75, 0.25], 
        subset(states, ['A', 'B'])
    )

    # P(C|A)
    fC_A = Factor(
        [0.8, 0.2, 0.1, 0.9], 
        subset(states, ['A', 'C'])
    )

    # Define a factor that holds the *conditional* distribution P(D|BC)
    fD_BC = Factor(
        [0.95, 0.05, 0.9, 0.1,0.8, 0.2, 0.0, 1.0], 
        subset(states, ['B', 'C', 'D'])
    )

    # P(E|C)
    fE_C = Factor(
        [0.7, 0.3, 0.0, 1.0], 
        subset(states, ['C', 'E'])
    )

