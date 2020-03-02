# -*- coding: utf-8 -*-
import unittest
import doctest
import logging
import itertools


from thomas.core.bayesiannetwork import BayesianNetwork, DiscreteNetworkNode
from thomas.core import examples

log = logging.getLogger(__name__)


class TestBayesianNetwork(unittest.TestCase):

    def setUp(self):
        self.Gs = examples.get_student_network()
        self.maxDiff = None

    def test_basic_setup(self):
        """Test instantiation of the Student network."""
        random_vars = ['D', 'I', 'G', 'S', 'L']

        for rv in random_vars:
            self.assertTrue(rv in self.Gs.nodes)
            self.assertTrue(isinstance(self.Gs.nodes[rv], DiscreteNetworkNode))

    def test_priors(self):
        """Test computation of a BN's prior probabilities for the nodes."""
        self.Gs.reset_evidence()
        priors = self.Gs.get_marginals()

        D = priors['D']
        self.assertAlmostEquals(D['d0'], 0.6)
        self.assertAlmostEquals(D['d1'], 0.4)
        self.assertAlmostEquals(D.sum(), 1)

        I = priors['I']
        self.assertAlmostEquals(I['i0'], 0.7)
        self.assertAlmostEquals(I['i1'], 0.3)
        self.assertAlmostEquals(I.sum(), 1)

        G = priors['G']
        self.assertAlmostEquals(G['g1'], 0.3620)
        self.assertAlmostEquals(G['g2'], 0.2884)
        self.assertAlmostEquals(G['g3'], 0.3496)
        self.assertAlmostEquals(G.sum(), 1)

        S = priors['S']
        self.assertAlmostEquals(S['s0'], 0.725)
        self.assertAlmostEquals(S['s1'], 0.275)
        self.assertAlmostEquals(S.sum(), 1)

        L = priors['L']
        self.assertAlmostEquals(L['l0'], 0.498, places=3)
        self.assertAlmostEquals(L['l1'], 0.502, places=3)
        self.assertAlmostEquals(L.sum(), 1)

    def test_compute_posterior_student(self):
        """Test computation of a BN's probabilities for the nodes given
        evidence.
        """
        places = 4

        def set_i0():
            self.Gs.reset_evidence()
            self.Gs.set_evidence_hard('I', 'i0')

            priors = self.Gs.get_marginals()

            I = priors['I']
            self.assertAlmostEquals(I['i0'], 1.0)
            self.assertAlmostEquals(I['i1'], 0.0)
            self.assertAlmostEquals(I.sum(), 1)

            D = priors['D']
            self.assertAlmostEquals(D['d0'], 0.6)
            self.assertAlmostEquals(D['d1'], 0.4)
            self.assertAlmostEquals(D.sum(), 1)

            G = priors['G']
            self.assertAlmostEquals(G['g1'], 0.20)
            self.assertAlmostEquals(G['g2'], 0.34)
            self.assertAlmostEquals(G['g3'], 0.46)
            self.assertAlmostEquals(G.sum(), 1)

            L = priors['L']
            self.assertAlmostEquals(L['l0'], 0.6114)
            self.assertAlmostEquals(L['l1'], 0.3886)
            self.assertAlmostEquals(L.sum(), 1)

            S = priors['S']
            self.assertAlmostEquals(S['s0'], 0.950)
            self.assertAlmostEquals(S['s1'], 0.050)
            self.assertAlmostEquals(S.sum(), 1)

        def add_d0():
            self.Gs.set_evidence_hard('D', 'd0')
            priors = self.Gs.get_marginals()

            I = priors['I']
            self.assertAlmostEquals(I['i0'], 1.0)
            self.assertAlmostEquals(I['i1'], 0.0)
            self.assertAlmostEquals(I.sum(), 1)

            D = priors['D']
            self.assertAlmostEquals(D['d0'], 1.0)
            self.assertAlmostEquals(D['d1'], 0.0)
            self.assertAlmostEquals(D.sum(), 1)

            G = priors['G']
            self.assertAlmostEquals(G['g1'], 0.30)
            self.assertAlmostEquals(G['g2'], 0.40)
            self.assertAlmostEquals(G['g3'], 0.30)
            self.assertAlmostEquals(G.sum(), 1)

            L = priors['L']
            self.assertAlmostEquals(L['l0'], 0.4870)
            self.assertAlmostEquals(L['l1'], 0.5130)
            self.assertAlmostEquals(L.sum(), 1)

            S = priors['S']
            self.assertAlmostEquals(S['s0'], 0.950)
            self.assertAlmostEquals(S['s1'], 0.050)
            self.assertAlmostEquals(S.sum(), 1)

        def reset_set_g1():
            self.Gs.reset_evidence()
            self.Gs.set_evidence_hard('G', 'g1')
            priors = self.Gs.get_marginals()

            I = priors['I']
            self.assertAlmostEquals(I['i0'], 0.3867, places=places)
            self.assertAlmostEquals(I['i1'], 0.6133, places=places)
            self.assertAlmostEquals(I.sum(), 1)

            D = priors['D']
            self.assertAlmostEquals(D['d0'], 0.7956, places=places)
            self.assertAlmostEquals(D['d1'], 0.2044, places=places)
            self.assertAlmostEquals(D.sum(), 1)

            G = priors['G']
            self.assertAlmostEquals(G['g1'], 1.0, places=places)
            self.assertAlmostEquals(G['g2'], 0.0, places=places)
            self.assertAlmostEquals(G['g3'], 0.0, places=places)
            self.assertAlmostEquals(G.sum(), 1)

            L = priors['L']
            self.assertAlmostEquals(L['l0'], 0.10, places=places)
            self.assertAlmostEquals(L['l1'], 0.90, places=places)
            self.assertAlmostEquals(L.sum(), 1)

            S = priors['S']
            self.assertAlmostEquals(S['s0'], 0.4901, places=places)
            self.assertAlmostEquals(S['s1'], 0.5099, places=places)
            self.assertAlmostEquals(S.sum(), 1)

        set_i0()
        add_d0()
        reset_set_g1()

    @unittest.skip
    def test_elimination_order_importance(self):
        self.Gs.reset_evidence()

        nodes = list(self.Gs.nodes.keys())

        for order in itertools.permutations(nodes):
            self.Gs._jt = None
            self.Gs.elimination_order = order

            priors = self.Gs.get_marginals()
            L = priors['L']
            self.assertAlmostEquals(L['l0'], 0.498, places=3)
            self.assertAlmostEquals(L['l1'], 0.502, places=3)

        self.Gs.elimination_order = None

    def test_compute_posterior_student_alt(self):
        """Test computation of a BN's probabilities for the nodes given
        evidence.
        """
        # P(I)
        I = self.Gs.compute_marginals(['I'])['I']
        self.assertAlmostEquals(I['i0'], 0.7, places=3)
        self.assertAlmostEquals(I['i1'], 0.3, places=3)

        # P(I=i1|L=l0)
        I_l0 = self.Gs.compute_marginals(['I'], {'L': 'l0'})['I']
        self.assertAlmostEquals(I_l0['i1'], 0.140, places=3)

        # P(I=i1|G=g3)
        I_g3 = self.Gs.compute_marginals(['I'], {'G': 'g3'})['I']
        self.assertAlmostEquals(I_g3['i1'], 0.079, places=3)

        # P(L=l1|I=i0, D=d0)
        L_i0d0 = self.Gs.compute_marginals(['L'], {'I': 'i0', 'D': 'd0'})['L']
        self.assertAlmostEquals(L_i0d0['l1'], 0.513, places=3)

        # P(I=i1|G=g3, S=s1)
        I_g3s1 = self.Gs.compute_marginals(['I'], {'G': 'g3', 'S': 's1'})['I']
        self.assertAlmostEquals(I_g3s1['i1'], 0.578, places=3)

        # Since I ⏊ L | G it follows that P(I|G, L=l0) == P(I|G, L=l1)
        # P(I=i1|G=g3, L=l0)
        I_g3l0 = self.Gs.compute_marginals(['I'], {'G': 'g3', 'L': 'l0'})['I']
        self.assertAlmostEquals(I_g3l0['i1'], 0.079, places=3)

        # P(I=i1|G=g3, L=l1)
        I_g3l1 = self.Gs.compute_marginals(['I'], {'G': 'g3', 'L': 'l1'})['I']
        self.assertAlmostEquals(I_g3l0['i1'], 0.079, places=3)

        # P(S|I=i1)
        S_i1 = self.Gs.compute_marginals(['S'], {'I': 'i1'})['S']
        self.assertAlmostEquals(S_i1['s0'], 0.2, places=3)
        self.assertAlmostEquals(S_i1['s1'], 0.8, places=3)

        # P(S|I=i0)
        S_i0 = self.Gs.compute_marginals(['S'], {'I': 'i0'})['S']
        self.assertAlmostEquals(S_i0['s0'], 0.95, places=3)
        self.assertAlmostEquals(S_i0['s1'], 0.05, places=3)

    def test_serialization(self):
        """Test serialization to and loading from dictionary."""
        serialized = self.Gs.as_dict()
        unserialized = BayesianNetwork.from_dict(serialized)

        self.assertDictEqual(serialized, unserialized.as_dict())


