# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
import unittest
import doctest
import logging

from thomas.core.bag import Bag
from thomas.core import examples

log = logging.getLogger(__name__)

# def load_tests(loader, tests, ignore):
#     # tests.addTests(doctest.DocTestSuite(...))
#     return tests

class TestBag(unittest.TestCase):

    def test_scope(self):
        """Test a Bag's scope."""
        # Get the Factors for the Sprinkler network
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        self.assertEquals(bag.scope, {'A', 'B', 'C', 'D', 'E'})

    def test_variable_elimination_single(self):
        """Test the variable elimination algorithm."""
        # Get the Factors for the Sprinkler network
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        # Compute the prior over C
        fC = bag.eliminate(['C'])

        self.assertAlmostEquals(fC['c0'], 0.48, places=2)
        self.assertAlmostEquals(fC['c1'], 0.52, places=2)
        self.assertAlmostEquals(fC.sum(), 1, places=8)

    def test_variable_elimination_multiple(self):
        """Test the variable elimination algorithm."""
        # Get the Factors for the Sprinkler network
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        # Compute the joint over A and C
        fAC = bag.eliminate(['A', 'C'])

        self.assertAlmostEquals(fAC['a0', 'c0'], 0.36, places=2)
        self.assertAlmostEquals(fAC['a0', 'c1'], 0.04, places=2)
        self.assertAlmostEquals(fAC['a1', 'c0'], 0.12, places=2)
        self.assertAlmostEquals(fAC['a1', 'c1'], 0.48, places=2)
        self.assertAlmostEquals(fAC.sum(), 1, places=8)

    def test_variable_elimination_with_evidence(self):
        """Test the variable elimination algorithm."""
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        # Compute the (unnormalized) factor over C and A=a1
        fC_a1 = bag.eliminate(['C'], {'A': 'a1'})

        self.assertAlmostEquals(fC_a1['c0'], 0.12, places=2)
        self.assertAlmostEquals(fC_a1['c1'], 0.48, places=2)

    def test_compute_posterior(self):
        """Test the function Bag.compute_posterior()."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        I = bag.compute_posterior(['I'], {}, [], {})
        self.assertAlmostEquals(I['i0'], 0.7, places=3)
        self.assertAlmostEquals(I['i1'], 0.3, places=3)

        S_i1 = bag.compute_posterior(['S'], {}, [], {'I': 'i1'})
        self.assertAlmostEquals(S_i1['s0'], 0.2, places=3)
        self.assertAlmostEquals(S_i1['s1'], 0.8, places=3)

        S_i0 = bag.compute_posterior(['S'], {}, [], {'I': 'i0'})
        self.assertAlmostEquals(S_i0['s0'], 0.95, places=3)
        self.assertAlmostEquals(S_i0['s1'], 0.05, places=3)

    def test_MAP(self):
        """Test the BayesianNetwork.MAP() function."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        argmax_I = bag.MAP(['I'], {}, False)
        self.assertEquals(argmax_I, 'i0')

        argmax_G = bag.MAP(['G'], {}, False)
        self.assertEquals(argmax_G, 'g1')

        argmax_G = bag.MAP(['G'], {}, True)
        self.assertEquals(argmax_G[0], 'g1')
        self.assertAlmostEquals(argmax_G[1], 0.362)

    def test_P(self):
        """Test the function BayesianNetwork.P()."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        I = bag.P('I')
        self.assertAlmostEquals(I['i0'], 0.7, places=3)
        self.assertAlmostEquals(I['i1'], 0.3, places=3)

        G_I = bag.P('G|I')
        self.assertEquals(G_I.scope, ['I', 'G'])
        self.assertEquals(G_I.conditioned, ['G'])
        self.assertEquals(G_I.conditioning, ['I'])

        I_g1 = bag.P('I|G=g1')
        self.assertAlmostEquals(I_g1['i0'], 0.387, places=3)
        self.assertAlmostEquals(I_g1['i1'], 0.613, places=3)

    def test_JPT(self):
        """Test Joint Probability Table."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        JPT = bag.eliminate(list(bag.scope)).normalize()
        self.assertEquals(JPT.sum(), 1)
