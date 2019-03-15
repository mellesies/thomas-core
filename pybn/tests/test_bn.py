# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
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

    def test_basic_setup(self):
        random_vars = ['D', 'I', 'G', 'S', 'L']

        for rv in random_vars:
            self.assertTrue(rv in self.Gs.nodes)
            self.assertTrue(isinstance(self.Gs.nodes[rv], bn.Node))

    def test_priors(self):
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

    def test_JPT(self):
        JPT = self.Gs.eliminate(self.Gs.scope)
        self.assertAlmostEquals(JPT.sum(), 1)

    