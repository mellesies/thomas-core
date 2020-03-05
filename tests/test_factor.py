# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function
import unittest
import doctest
import logging

from thomas.core.factor import Factor
from thomas.core import examples

log = logging.getLogger(__name__)

# def load_tests(loader, tests, ignore):
#     # tests.addTests(doctest.DocTestSuite(...))
#     return tests

class TestFactor(unittest.TestCase):

    def test_state_order(self):
        """Test that a Factor's (Multi)Index keeps its states in order.

            Regression test for GitHub issue #1.
        """
        # P(A)
        fA = Factor(
            [0.6, 0.4],
            {'A': ['a1', 'a0']}
        )

        self.assertEquals(fA['a1'], 0.6)
        self.assertEquals(fA['a0'], 0.4)

        # P(B|A)
        fB_A = Factor(
            [0.2, 0.8, 0.75, 0.25],
            {'A': ['a1', 'a0'],'B': ['b1', 'b0']}
        )

        self.assertEquals(fB_A['a1', 'b1'], 0.20)
        self.assertEquals(fB_A['a1', 'b0'], 0.80)
        self.assertEquals(fB_A['a0', 'b1'], 0.75)
        self.assertEquals(fB_A['a0', 'b0'], 0.25)

    def test_multiplication(self):
        """Test factor multiplication."""
        # Get the Factors for the Sprinkler network
        fA, fB_A, fC_A, fD_BC, fE_C = examples.get_sprinkler_factors()

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
        fA, fB_A, fC_A, fD_BC, fE_C = examples.get_sprinkler_factors()

        # Multiplying the factor with a *prior* with a *conditional* distribution, yields
        # a *joint* distribution.
        fAB = fA * fB_A

        # By summing out A, we'll get the prior over B
        fB = fAB.sum_out('A')

        # Make sure we did this right :-)
        self.assertAlmostEquals(fB['b1'], 0.42, places=2)
        self.assertAlmostEquals(fB['b0'], 0.58, places=2)

        self.assertAlmostEquals(fB.sum(), 1, places=8)

    def test_summing_out_all(self):
        """Test summing out variables."""
        # Get the Factors for the Sprinkler network
        fA, fB_A, fC_A, fD_BC, fE_C = examples.get_sprinkler_factors()

        # Multiplying the factor with a *prior* with a *conditional* distribution, yields
        # a *joint* distribution.
        fAB = fA * fB_A

        # By summing out A, we'll get the prior over B
        total = fAB.sum_out(['A', 'B'])

        # Make sure we did this right :-)
        self.assertAlmostEquals(total, 1.00, places=2)

    def test_project(self):
        """Test the `project` function."""
        fA, fB_A, fC_B = examples.get_example7_factors()
        fAfB = fA * fB_A
        fBfC = fAfB.sum_out('A') * fC_B

        fC = fBfC.project({'C'})
        self.assertAlmostEquals(fC['c0'], 0.624, places=8)
        self.assertAlmostEquals(fC['c1'], 0.376, places=8)
        self.assertAlmostEquals(fC.sum(), 1, places=8)

    def test_serialization_simple(self):
        """Test the JSON serialization."""
        [fA, fB_A, fC_A, fD_BC, fE_C] = examples.get_sprinkler_factors()

        dict_repr = fA.as_dict()

        # Make sure the expected keys exist
        for key in ['type', 'scope', 'variable_states', 'data']:
            self.assertTrue(key in dict_repr)

        self.assertTrue(dict_repr['type'] == 'Factor')

        fA2 = Factor.from_dict(dict_repr)
        self.assertEquals(fA.scope, fA2.scope)
        self.assertEquals(fA.variable_states, fA2.variable_states)

    def test_serialization_complex(self):
        """Test the JSON serialization."""
        [fA, fB_A, fC_A, fD_BC, fE_C] = examples.get_sprinkler_factors()

        dict_repr = fB_A.as_dict()

        # Make sure the expected keys exist
        for key in ['type', 'scope', 'variable_states', 'data']:
            self.assertTrue(key in dict_repr)

        self.assertTrue(dict_repr['type'] == 'Factor')

        fB_A2 = Factor.from_dict(dict_repr)
        self.assertEquals(fB_A.scope, fB_A2.scope)
        self.assertEquals(fB_A.variable_states, fB_A2.variable_states)

