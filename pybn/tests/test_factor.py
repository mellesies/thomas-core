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

class TestFactor(unittest.TestCase):
    
    def test_multiplication(self):
        """Test factor multiplication."""
        # Get the Factors for the Sprinkler network
        fA, fB_A, fC_A, fD_BC, fE_C = pybn.examples.get_sprinkler_factors()

        # Multiplying the factor with a *prior* with a *conditional* distribution, yields
        # a *joint* distribution.
        fAB = fA * fB_A

        # Make sure we did this right :-)
        self.assertAlmostEquals(fAB['a1', 'b1'], 0.12, places=2)
        self.assertAlmostEquals(fAB['a1', 'b0'], 0.48, places=2)
        self.assertAlmostEquals(fAB['a0', 'b1'], 0.30, places=2)
        self.assertAlmostEquals(fAB['a0', 'b0'], 0.10, places=2)

        self.assertAlmostEquals(fAB.sum(), 1, places=8)

    def test_summing_out(self):
        """Test summing out variables."""
        # Get the Factors for the Sprinkler network
        fA, fB_A, fC_A, fD_BC, fE_C = pybn.examples.get_sprinkler_factors()

        # Multiplying the factor with a *prior* with a *conditional* distribution, yields
        # a *joint* distribution.
        fAB = fA * fB_A

        # By summing out A, we'll get the prior over B
        fB = fAB.sum_out('A')

        # Make sure we did this right :-)
        self.assertAlmostEquals(fB['b1'], 0.42, places=2)
        self.assertAlmostEquals(fB['b0'], 0.58, places=2)

        self.assertAlmostEquals(fB.sum(), 1, places=8)

    def test_serialization_simple(self):
        """Test the JSON serialization."""
        [fA, fB_A, fC_A, fD_BC, fE_C] = bn.examples.get_sprinkler_factors()

        dict_repr = fA.as_dict()

        # Make sure the expected keys exist
        for key in ['type', 'scope', 'variable_states', 'data']:
            self.assertTrue(key in dict_repr)

        self.assertTrue(dict_repr['type'] == 'Factor')

        fA2 = bn.Factor.from_dict(dict_repr)
        self.assertEquals(fA.scope, fA2.scope)
        self.assertEquals(fA.variable_states, fA2.variable_states)

    def test_serialization_complex(self):
        """Test the JSON serialization."""
        [fA, fB_A, fC_A, fD_BC, fE_C] = bn.examples.get_sprinkler_factors()

        dict_repr = fB_A.as_dict()

        # Make sure the expected keys exist
        for key in ['type', 'scope', 'variable_states', 'data']:
            self.assertTrue(key in dict_repr)

        self.assertTrue(dict_repr['type'] == 'Factor')
        
        fB_A2 = bn.Factor.from_dict(dict_repr)
        self.assertEquals(fB_A.scope, fB_A2.scope)
        self.assertEquals(fB_A.variable_states, fB_A2.variable_states)

