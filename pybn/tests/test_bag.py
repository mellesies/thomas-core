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

class TestBag(unittest.TestCase):
    
    def test_scope(self):
        """Test a Bag's scope."""
        # Get the Factors for the Sprinkler network
        factors = pybn.examples.get_sprinkler_factors()        
        bag = bn.Bag('Sprinkler', factors)

        self.assertEquals(bag.scope, {'A', 'B', 'C', 'D', 'E'})

    def test_variable_elimination_single(self):
        """Test the variable elimination algorithm."""
        # Get the Factors for the Sprinkler network
        factors = pybn.examples.get_sprinkler_factors()        
        bag = bn.Bag('Sprinkler', factors)

        # Compute the prior over C
        fC = bag.eliminate(['C'])

        self.assertAlmostEquals(fC['c0'], 0.48, places=2)
        self.assertAlmostEquals(fC['c1'], 0.52, places=2)
        self.assertAlmostEquals(fC.sum(), 1, places=8)

    def test_variable_elimination_multiple(self):
        """Test the variable elimination algorithm."""
        # Get the Factors for the Sprinkler network
        factors = pybn.examples.get_sprinkler_factors()        
        bag = bn.Bag('Sprinkler', factors)

        # Compute the joint over A and C
        fAC = bag.eliminate(['A', 'C'])

        self.assertAlmostEquals(fAC['a0', 'c0'], 0.36, places=2)
        self.assertAlmostEquals(fAC['a0', 'c1'], 0.04, places=2)
        self.assertAlmostEquals(fAC['a1', 'c0'], 0.12, places=2)
        self.assertAlmostEquals(fAC['a1', 'c1'], 0.48, places=2)
        self.assertAlmostEquals(fAC.sum(), 1, places=8)


    def test_variable_elimination_with_evidence(self):
        """Test the variable elimination algorithm."""
        factors = pybn.examples.get_sprinkler_factors()        
        bag = bn.Bag('Sprinkler', factors)

        # Compute the (unnormalized) factor over C and A=a1
        fC_a1 = bag.eliminate(['C'], {'A': 'a1'})

        self.assertAlmostEquals(fC_a1['c0'], 0.12, places=2)
        self.assertAlmostEquals(fC_a1['c1'], 0.48, places=2)

