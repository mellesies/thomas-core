# -*- coding: utf-8 -*-
import os
import unittest
import doctest
import logging
import itertools
import json

from tempfile import gettempdir

import pandas as pd
import numpy as np

import thomas.core
from thomas.core.cpt import CPT
from thomas.core.bayesiannetwork import BayesianNetwork, DiscreteNetworkNode
from thomas.core import examples


log = logging.getLogger(__name__)

class TestBayesianNetwork(unittest.TestCase):

    def setUp(self):
        thomas.core.options['quiet'] = True
        self.Gs = examples.get_student_network()
        self.maxDiff = None

    def test_basic_setup(self):
        """Test instantiation of the Student network."""
        random_vars = ['D', 'I', 'G', 'S', 'L']

        for rv in random_vars:
            self.assertTrue(rv in self.Gs.nodes)
            self.assertIsInstance(self.Gs.nodes[rv], DiscreteNetworkNode)

    def test_repr(self):
        """Test repr(bn)."""
        n = self.Gs.name
        self.assertTrue(repr(self.Gs).startswith(f"<BayesianNetwork name='{n}'>"))

    def test_states(self):
        """Test property bn.states."""

        # The index for states should be the same keys as the one for the nodes.
        self.assertEqual(self.Gs.states.keys(), self.Gs.nodes.keys())

    def test_node_repr(self):
        """Test repr(node)."""
        node = self.Gs['G']
        self.assertTrue(repr(node).startswith('DiscreteNetworkNode'))

    def test_node_parents(self):
        """Test property Node.parents."""
        node = self.Gs['G']
        self.assertEqual(len(node.parents), 2)
        parent_RVs = set([p.RV for p in node.parents])
        self.assertEqual(parent_RVs, {'I', 'D'})

    def test_node_cpt(self):
        """Test getting/setting node's CPT."""
        node = self.Gs['I']
        cpts = examples.get_student_CPTs()

        with self.assertRaises(Exception):
            node.cpt = 1

        with self.assertRaises(Exception):
            node.cpt = CPT(cpts['G'].as_factor(), conditioned=['I', 'D'])

        with self.assertRaises(Exception):
            node.cpt = CPT(cpts['G'])

    def test_node_reset(self):
        """Test node.reset()."""
        Gs = examples.get_student_network()
        Gs['I'].reset()

        self.assertAlmostEqual(Gs['I'].cpt['i0'], 0.5)

    def test_node_validate(self):
        """Test node.validate()."""
        try:
            self.Gs['I'].validate()
        except Exception:
            self.fail("validate() raised Exception unexpectedly!")

    def test_get_marginals_no_evidence(self):
        """Test computation of a BN's prior probabilities for the nodes."""
        self.Gs.reset_evidence()
        priors = self.Gs.get_marginals()

        D = priors['D']
        self.assertAlmostEqual(D['d0'], 0.6)
        self.assertAlmostEqual(D['d1'], 0.4)
        self.assertAlmostEqual(D.sum(), 1)

        I = priors['I']
        self.assertAlmostEqual(I['i0'], 0.7)
        self.assertAlmostEqual(I['i1'], 0.3)
        self.assertAlmostEqual(I.sum(), 1)

        G = priors['G']
        self.assertAlmostEqual(G['g1'], 0.3620)
        self.assertAlmostEqual(G['g2'], 0.2884)
        self.assertAlmostEqual(G['g3'], 0.3496)
        self.assertAlmostEqual(G.sum(), 1)

        S = priors['S']
        self.assertAlmostEqual(S['s0'], 0.725)
        self.assertAlmostEqual(S['s1'], 0.275)
        self.assertAlmostEqual(S.sum(), 1)

        L = priors['L']
        self.assertAlmostEqual(L['l0'], 0.498, places=3)
        self.assertAlmostEqual(L['l1'], 0.502, places=3)
        self.assertAlmostEqual(L.sum(), 1)

    def test_get_marginals_with_evidence(self):
        """Test computation of a BN's probabilities for the nodes given
        evidence.
        """
        places = 4

        def set_i0():
            self.Gs.reset_evidence()
            self.Gs.set_evidence_hard('I', 'i0')

            priors = self.Gs.get_marginals()

            I = priors['I']
            self.assertAlmostEqual(I['i0'], 1.0)
            self.assertAlmostEqual(I['i1'], 0.0)
            self.assertAlmostEqual(I.sum(), 1)

            D = priors['D']
            self.assertAlmostEqual(D['d0'], 0.6)
            self.assertAlmostEqual(D['d1'], 0.4)
            self.assertAlmostEqual(D.sum(), 1)

            G = priors['G']
            self.assertAlmostEqual(G['g1'], 0.20)
            self.assertAlmostEqual(G['g2'], 0.34)
            self.assertAlmostEqual(G['g3'], 0.46)
            self.assertAlmostEqual(G.sum(), 1)

            L = priors['L']
            self.assertAlmostEqual(L['l0'], 0.6114)
            self.assertAlmostEqual(L['l1'], 0.3886)
            self.assertAlmostEqual(L.sum(), 1)

            S = priors['S']
            self.assertAlmostEqual(S['s0'], 0.950)
            self.assertAlmostEqual(S['s1'], 0.050)
            self.assertAlmostEqual(S.sum(), 1)

        def add_d0():
            self.Gs.set_evidence_hard('D', 'd0')
            priors = self.Gs.get_marginals()

            I = priors['I']
            self.assertAlmostEqual(I['i0'], 1.0)
            self.assertAlmostEqual(I['i1'], 0.0)
            self.assertAlmostEqual(I.sum(), 1)

            D = priors['D']
            self.assertAlmostEqual(D['d0'], 1.0)
            self.assertAlmostEqual(D['d1'], 0.0)
            self.assertAlmostEqual(D.sum(), 1)

            G = priors['G']
            self.assertAlmostEqual(G['g1'], 0.30)
            self.assertAlmostEqual(G['g2'], 0.40)
            self.assertAlmostEqual(G['g3'], 0.30)
            self.assertAlmostEqual(G.sum(), 1)

            L = priors['L']
            self.assertAlmostEqual(L['l0'], 0.4870)
            self.assertAlmostEqual(L['l1'], 0.5130)
            self.assertAlmostEqual(L.sum(), 1)

            S = priors['S']
            self.assertAlmostEqual(S['s0'], 0.950)
            self.assertAlmostEqual(S['s1'], 0.050)
            self.assertAlmostEqual(S.sum(), 1)

        def reset_set_g1():
            self.Gs.reset_evidence()
            self.Gs.set_evidence_hard('G', 'g1')
            priors = self.Gs.get_marginals()

            I = priors['I']
            self.assertAlmostEqual(I['i0'], 0.3867, places=places)
            self.assertAlmostEqual(I['i1'], 0.6133, places=places)
            self.assertAlmostEqual(I.sum(), 1)

            D = priors['D']
            self.assertAlmostEqual(D['d0'], 0.7956, places=places)
            self.assertAlmostEqual(D['d1'], 0.2044, places=places)
            self.assertAlmostEqual(D.sum(), 1)

            G = priors['G']
            self.assertAlmostEqual(G['g1'], 1.0, places=places)
            self.assertAlmostEqual(G['g2'], 0.0, places=places)
            self.assertAlmostEqual(G['g3'], 0.0, places=places)
            self.assertAlmostEqual(G.sum(), 1)

            L = priors['L']
            self.assertAlmostEqual(L['l0'], 0.10, places=places)
            self.assertAlmostEqual(L['l1'], 0.90, places=places)
            self.assertAlmostEqual(L.sum(), 1)

            S = priors['S']
            self.assertAlmostEqual(S['s0'], 0.4901, places=places)
            self.assertAlmostEqual(S['s1'], 0.5099, places=places)
            self.assertAlmostEqual(S.sum(), 1)

        set_i0()
        add_d0()
        reset_set_g1()

    def test_compute_marginals(self):
        """Test computation of a BN's probabilities for the nodes given
        evidence.
        """
        # P(I)
        I = self.Gs.compute_marginals(['I'])['I']
        self.assertAlmostEqual(I['i0'], 0.7, places=3)
        self.assertAlmostEqual(I['i1'], 0.3, places=3)

        # P(I=i1|L=l0)
        I_l0 = self.Gs.compute_marginals(['I'], {'L': 'l0'})['I']
        self.assertAlmostEqual(I_l0['i1'], 0.140, places=3)

        # P(I=i1|G=g3)
        I_g3 = self.Gs.compute_marginals(['I'], {'G': 'g3'})['I']
        self.assertAlmostEqual(I_g3['i1'], 0.079, places=3)

        # P(L=l1|I=i0, D=d0)
        L_i0d0 = self.Gs.compute_marginals(['L'], {'I': 'i0', 'D': 'd0'})['L']
        self.assertAlmostEqual(L_i0d0['l1'], 0.513, places=3)

        # P(I=i1|G=g3, S=s1)
        I_g3s1 = self.Gs.compute_marginals(['I'], {'G': 'g3', 'S': 's1'})['I']
        self.assertAlmostEqual(I_g3s1['i1'], 0.578, places=3)

        # Since I ⏊ L | G it follows that P(I|G, L=l0) == P(I|G, L=l1)
        # P(I=i1|G=g3, L=l0)
        I_g3l0 = self.Gs.compute_marginals(['I'], {'G': 'g3', 'L': 'l0'})['I']
        self.assertAlmostEqual(I_g3l0['i1'], 0.079, places=3)

        # P(I=i1|G=g3, L=l1)
        I_g3l1 = self.Gs.compute_marginals(['I'], {'G': 'g3', 'L': 'l1'})['I']
        self.assertAlmostEqual(I_g3l0['i1'], 0.079, places=3)

        # P(S|I=i1)
        S_i1 = self.Gs.compute_marginals(['S'], {'I': 'i1'})['S']
        self.assertAlmostEqual(S_i1['s0'], 0.2, places=3)
        self.assertAlmostEqual(S_i1['s1'], 0.8, places=3)

        # P(S|I=i0)
        S_i0 = self.Gs.compute_marginals(['S'], {'I': 'i0'})['S']
        self.assertAlmostEqual(S_i0['s0'], 0.95, places=3)
        self.assertAlmostEqual(S_i0['s1'], 0.05, places=3)

    def test_compute_posterior(self):
        """Test bn.compute_posterior()."""
        # P(D,S), can be computed with the default JT
        ID = self.Gs.compute_posterior(['I', 'D'], {}, [], {})
        self.assertAlmostEqual(ID.sum(), 1, places=3)


        I_D = self.Gs.compute_posterior(['I'], {}, ['D'], {})
        self.assertAlmostEqual(I_D.sum(), 2, places=3)

        # I and D are independent
        self.assertTrue(np.allclose(I_D['d0'], I_D['d1']))

        # P(D,S), cannot be computed with the default JT
        DS = self.Gs.compute_posterior(['D', 'S'], {}, [], {})
        self.assertAlmostEqual(DS.sum(), 1, places=3)

        # This fails ...
        # s0 = self.Gs.compute_posterior([], {'S': 's0'}, [], {})

    def test_joint_with_jt(self):
        """Testing computing a joint using a JT."""
        IS1 = self.Gs.compute_joint_with_jt(['I', 'S'])
        IS2 = self.Gs.compute_posterior(
            ['I', 'S'], {}, [], {},
            use_VE=True
        )

        self.assertTrue(IS1.equals(IS2))

    def test_ML_estimation(self):
        """Test ML estimation using a simple dataset."""

        # Create the nodes
        H = DiscreteNetworkNode('H', states=['T', 'F'])
        S = DiscreteNetworkNode('S', states=['T', 'F'])
        E = DiscreteNetworkNode('E', states=['T', 'F'])

        nodes = [H, S, E]

        # Create the edges
        edges = [
            ('H', 'S'),
            ('H', 'E'),
        ]

        # Create the BN
        bn = BayesianNetwork('Example 17.2', nodes, edges)

        # Load the data
        filename = thomas.core.get_pkg_data('dataset_17_2.csv')
        df = pd.read_csv(filename, sep=';')

        # Do the learning
        bn.ML_estimation(df)

        # Check the result
        self.assertAlmostEqual(bn['H'].cpt['F'], 1/4)
        self.assertAlmostEqual(bn['H'].cpt['T'], 3/4)

        self.assertAlmostEqual(bn['S'].cpt['T', 'T'], 1/6)
        self.assertAlmostEqual(bn['S'].cpt['T', 'F'], 5/6)
        self.assertAlmostEqual(bn['S'].cpt['F', 'T'], 1/4)
        self.assertAlmostEqual(bn['S'].cpt['F', 'F'], 3/4)

        self.assertAlmostEqual(bn['E'].cpt['T', 'T'], 11/12)
        self.assertAlmostEqual(bn['E'].cpt['T', 'F'], 1/12)
        self.assertAlmostEqual(bn['E'].cpt['F', 'T'], 1/2)
        self.assertAlmostEqual(bn['E'].cpt['F', 'F'], 1/2)

    def test_serialization(self):
        """Test serialization to and loading from dictionary."""
        serialized = self.Gs.as_dict()
        unserialized = BayesianNetwork.from_dict(serialized)

        self.assertDictEqual(serialized, unserialized.as_dict())

        tmpfile = os.path.join(gettempdir(), 'network.json')
        self.Gs.save(tmpfile)

        opened = BayesianNetwork.open(tmpfile)
        self.assertDictEqual(serialized, opened.as_dict())

        # Clean up
        os.remove(tmpfile)

    def test_elimination_order_importance(self):
        self.Gs.reset_evidence()

        nodes = list(self.Gs.nodes.keys())

        for order in itertools.permutations(nodes):
            self.Gs._jt = None
            self.Gs.elimination_order = order

            priors = self.Gs.get_marginals()
            L = priors['L']
            self.assertAlmostEqual(L['l0'], 0.498, places=3)
            self.assertAlmostEqual(L['l1'], 0.502, places=3)

        self.Gs.elimination_order = None

    @unittest.skip('Not yet')
    def test_EM_learning(self):
        """Test the EM-learning algorithm."""
        # Load the BN (with priors)
        bn = examples.get_example17_3_network()

        # Load the data
        filename = thomas.core.get_pkg_data('dataset_17_3.csv')
        df = pd.read_csv(filename, sep=';').set_index('Case')

        # Basic sanity check
        self.assertEqual(df.isna().sum().sum(), 8)
        self.assertEqual(bn['A'].cpt['a1'], 0.2)
        self.assertEqual(bn['A'].cpt['a2'], 0.8)

        # Run EM-learning. This should update the CPTs in place.
        bn.EM_learning(df, max_iterations=1)

        self.assertAlmostEqual(bn['A'].cpt['a1'], 0.421, places=3)
        self.assertAlmostEqual(bn['A'].cpt['a2'], 0.579, places=3)

        self.assertAlmostEqual(bn['B'].cpt['a1', 'b1'], 0.884, places=3)
        self.assertAlmostEqual(bn['B'].cpt['a1', 'b2'], 0.116, places=3)
        self.assertAlmostEqual(bn['B'].cpt['a2', 'b1'], 0.394, places=3)
        self.assertAlmostEqual(bn['B'].cpt['a2', 'b2'], 0.606, places=3)

        self.assertAlmostEqual(bn['C'].cpt['a1', 'c1'], 0.426, places=3)
        self.assertAlmostEqual(bn['C'].cpt['a1', 'c2'], 0.574, places=3)
        self.assertAlmostEqual(bn['C'].cpt['a2', 'c1'], 0.666, places=3)
        self.assertAlmostEqual(bn['C'].cpt['a2', 'c2'], 0.334, places=3)

        self.assertAlmostEqual(bn['D'].cpt['b1', 'd1'], 0.067, places=3)
        self.assertAlmostEqual(bn['D'].cpt['b1', 'd2'], 0.933, places=3)
        self.assertAlmostEqual(bn['D'].cpt['b2', 'd1'], 1.000, places=3)
        self.assertAlmostEqual(bn['D'].cpt['b2', 'd2'], 0.000, places=3)
