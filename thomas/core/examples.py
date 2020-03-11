# -*- coding: utf-8 -*-
"""Example Bayesian networks."""
import os

import numpy as np
import pandas as pd

import thomas.core
import thomas.core.reader.oobn

from .factor import Factor
from .cpt import CPT
from .jpt import JPT

from .bayesiannetwork import BayesianNetwork, DiscreteNetworkNode

def subset(full_dict, keys):
    """Return a subset of a dict."""
    return {k: full_dict[k] for k in keys}

def get_student_CPTs():
    """Return the CPTs for the Student Bayesian Network."""
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
        variable_states=subset(states, ['I']),
        description='Intelligence'
    )

    P['S'] = CPT(
        [0.95, 0.05,
         0.20, 0.80],
        variable_states=subset(states, ['I', 'S']),
        description='SAT Score'
    )

    P['D'] = CPT(
        [0.6, 0.4],
        variable_states=subset(states, ['D']),
        description='Difficulty'
    )

    P['G'] = CPT(
        [0.30, 0.40, 0.30,
         0.05, 0.25, 0.70,
         0.90, 0.08, 0.02,
         0.50, 0.30, 0.20],
        variable_states=subset(states, ['I', 'D', 'G']),
        description='Grade'
    )

    P['L'] = CPT(
        [0.10, 0.90,
         0.40, 0.60,
         0.99, 0.01],
        variable_states=subset(states, ['G', 'L']),
        description='Letter'
    )

    return P

def get_student_network_from_CPTs():
    """Return the Student Bayesian Network."""
    P = get_student_CPTs()
    return BayesianNetwork.from_CPTs('Student', P.values())

def get_student_network():
    """Return the Student Bayesian Network."""
    filename = thomas.core.get_pkg_data('student.json')
    return BayesianNetwork.open(filename)

def get_sprinkler_factors():
    """Return the factors for the Sprinkler Bayesian Network.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 127-128) by Adnan Darwiche (2009).
    """
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
        [0.95, 0.05, 0.9, 0.1, 0.8, 0.2, 0.0, 1.0],
        subset(states, ['B', 'C', 'D'])
    )

    # P(E|C)
    fE_C = Factor(
        [0.7, 0.3, 0.0, 1.0],
        subset(states, ['C', 'E'])
    )

    return [fA, fB_A, fC_A, fD_BC, fE_C]

def get_sprinkler_jpt():
    """Return the JPT for the Sprinkler network.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 127-128) by Adnan Darwiche (2009).
    """
    states = {
        'A': ['a1', 'a0'],
        'B': ['b1', 'b0'],
        'C': ['c1', 'c0'],
        'D': ['d1', 'd0'],
        'E': ['e1', 'e0'],
    }

    data = [0.06384, 0.02736, 0.00336, 0.00144, 0, 0.02160,
            0, 0.00240, 0.21504, 0.09216, 0.05376, 0.02304,
            0, 0, 0, 0.09600, 0.01995, 0.00855, 0.00105,
            0.00045, 0, 0.24300, 0, 0.02700, 0.00560, 0.00240,
            0.00140, 0.00060, 0, 0, 0, 0.090]

    return JPT(data, states)

def get_sprinkler_network_from_factors():
    """Return the Sprinkler Bayesian Network."""
    factors = get_sprinkler_factors()
    CPTs = [CPT(f) for f in factors]
    return BayesianNetwork.from_CPTs('Sprinkler', CPTs)


def get_sprinkler_network():
    """Return the Sprinkler Network."""
    filename = thomas.core.get_pkg_data('sprinkler.json')
    return BayesianNetwork.open(filename)

def get_example7_factors():
    """Return the factors for a very simple BN.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 154) by Adnan Darwiche (2009).
    """
    states = {
        'A': ['a1', 'a0'],
        'B': ['b1', 'b0'],
        'C': ['c1', 'c0'],
    }

    # P(A)
    fA = Factor(
        [0.6, 0.4],
        subset(states, ['A'])
    )

    # P(B|A)
    fB_A = Factor(
        [0.9, 0.1, 0.2, 0.8],
        subset(states, ['A', 'B'])
    )

    # P(C|A)
    fC_B = Factor(
        [0.3, 0.7, 0.5, 0.5],
        subset(states, ['B', 'C'])
    )

    return fA, fB_A, fC_B

def get_example7_network():
    """Return a very simple BN.

    Data copied from "Modeling and Reasoning with Bayesian Networks"
    (page 154) by Adnan Darwiche (2009).
    """
    fA, fB_A, fC_B = get_example7_factors()
    nA = DiscreteNetworkNode('A', fA)
    nB = DiscreteNetworkNode('B', fB_A)
    nC = DiscreteNetworkNode('C', fC_B)

    bn = BayesianNetwork(
        'class',
        [nA, nB, nC],
        [('A', 'B'), ('B', 'C')]
    )

    return bn

def get_lungcancer_network():
    """Load 'lungcancer.oobn'."""
    filename = thomas.core.get_pkg_data('lungcancer.oobn')
    return thomas.core.reader.oobn.read(filename)