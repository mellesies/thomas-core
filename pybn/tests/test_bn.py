# -*- coding: utf-8 -*-
import unittest
import doctest
import logging

import pybn as bn
import pybn.examples

log = logging.getLogger(__name__)

# def load_tests(loader, tests, ignore):
#     # tests.addTests(doctest.DocTestSuite(...))
#     return tests

class TestBayesianNetwork(unittest.TestCase):
    
    def setUp(self):
        self.Gs = bn.examples.get_student_network()
        self.maxDiff = None

    def test_basic_setup(self):
        random_vars = ['D', 'I', 'G', 'S', 'L']

        for rv in random_vars:
            self.assertTrue(rv in self.Gs.nodes)
            self.assertTrue(isinstance(self.Gs.nodes[rv], bn.Node))

    def test_priors(self):
        """Test computation of a BN's prior probabilities for the nodes."""
        D = self.Gs.eliminate(['D']).normalize()
        self.assertAlmostEquals(D['d0'], 0.6)        
        self.assertAlmostEquals(D['d1'], 0.4)
        self.assertAlmostEquals(D.sum(), 1)

        I = self.Gs.eliminate(['I']).normalize()
        self.assertAlmostEquals(I['i0'], 0.7)
        self.assertAlmostEquals(I['i1'], 0.3)
        self.assertAlmostEquals(I.sum(), 1)

        G = self.Gs.eliminate(['G']).normalize()
        self.assertAlmostEquals(G['g1'], 0.3620)
        self.assertAlmostEquals(G['g2'], 0.2884)
        self.assertAlmostEquals(G['g3'], 0.3496)
        self.assertAlmostEquals(G.sum(), 1)

        S = self.Gs.eliminate(['S']).normalize()
        self.assertAlmostEquals(S['s0'], 0.725)        
        self.assertAlmostEquals(S['s1'], 0.275)
        self.assertAlmostEquals(S.sum(), 1)

        L = self.Gs.eliminate(['L']).normalize()
        self.assertAlmostEquals(L['l0'], 0.498, places=3)
        self.assertAlmostEquals(L['l1'], 0.502, places=3)
        self.assertAlmostEquals(L.sum(), 1)

    def test_posteriors(self):
        """Test computation of a BN's probabilities for the nodes given 
        evidence.
        """
        # P(I=i1|L=l0)
        I_l0 = self.Gs.eliminate(['I'], {'L': 'l0'}).normalize()
        self.assertAlmostEquals(I_l0['i1'], 0.140, places=3)

        # P(I=i1|G=g3)
        I_g3 = self.Gs.eliminate(['I'], {'G': 'g3'}).normalize()
        self.assertAlmostEquals(I_g3['i1'], 0.079, places=3)

        # P(L=l1|I=i0, D=d0)
        L_i0d0 = self.Gs.eliminate(['L'], {'I': 'i0', 'D': 'd0'}).normalize()
        self.assertAlmostEquals(L_i0d0['l1'], 0.513, places=3)

        # P(I=i1|G=g3, S=s1)
        I_g3s1 = self.Gs.eliminate(['I'], {'G': 'g3', 'S': 's1'}).normalize()
        self.assertAlmostEquals(I_g3s1['i1'], 0.578, places=3)

        # Since I ⏊ L | G it follows that P(I|G, L=l0) == P(I|G, L=l1)
        # P(I=i1|G=g3, L=l0)
        I_g3l0 = self.Gs.eliminate(['I'], {'G': 'g3', 'L': 'l0'}).normalize()
        self.assertAlmostEquals(I_g3l0['i1'], 0.079, places=3)

        # P(I=i1|G=g3, L=l1)
        I_g3l1 = self.Gs.eliminate(['I'], {'G': 'g3', 'L': 'l1'}).normalize()
        self.assertAlmostEquals(I_g3l0['i1'], 0.079, places=3)

    def test_compute_posterior(self):
        """Test the function BayesianNetwork.compute_posterior()."""
        I = self.Gs.compute_posterior(['I'], {}, [], {})
        self.assertAlmostEquals(I['i0'], 0.7, places=3)
        self.assertAlmostEquals(I['i1'], 0.3, places=3)

        S_i1 = self.Gs.compute_posterior(['S'], {}, [], {'I': 'i1'})
        self.assertAlmostEquals(S_i1['s0'], 0.2, places=3)
        self.assertAlmostEquals(S_i1['s1'], 0.8, places=3)

        S_i0 = self.Gs.compute_posterior(['S'], {}, [], {'I': 'i0'})
        self.assertAlmostEquals(S_i0['s0'], 0.95, places=3)
        self.assertAlmostEquals(S_i0['s1'], 0.05, places=3)

    def test_MAP(self):
        """Test the BayesianNetwork.MAP() function."""
        argmax_I = self.Gs.MAP(['I'], {}, False)
        self.assertEquals(argmax_I, 'i0')

        argmax_G = self.Gs.MAP(['G'], {}, False)
        self.assertEquals(argmax_G, 'g1')

        argmax_G = self.Gs.MAP(['G'], {}, True)
        self.assertEquals(argmax_G[0], 'g1')
        self.assertAlmostEquals(argmax_G[1], 0.362)

    def test_P(self):
        """Test the function BayesianNetwork.P()."""
        I = self.Gs.P('I')
        self.assertAlmostEquals(I['i0'], 0.7, places=3)
        self.assertAlmostEquals(I['i1'], 0.3, places=3)

        G_I = self.Gs.P('G|I')
        self.assertEquals(G_I.scope, ['I', 'G'])
        self.assertEquals(G_I.conditioned, ['G'])
        self.assertEquals(G_I.conditioning, ['I'])

        I_g1 = self.Gs.P('I|G=g1')
        self.assertAlmostEquals(I_g1['i0'], 0.387, places=3)
        self.assertAlmostEquals(I_g1['i1'], 0.613, places=3)

    def test_serialization(self):
        """Test serialization to and loading from dictionary."""
        serialized = self.Gs.as_dict()
        unserialized = bn.BayesianNetwork.from_dict(serialized)

        self.assertDictEqual(serialized, unserialized.as_dict())

    def test_JPT(self):
        """Test Joint Probability Table."""
        JPT = self.Gs.eliminate(list(self.Gs.scope)).normalize()
        self.assertEquals(JPT.sum(), 1)
