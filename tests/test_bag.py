# -*- coding: utf-8 -*-
import unittest
import doctest
import logging

from thomas.core.bag import Bag
from thomas.core import examples

log = logging.getLogger(__name__)

class TestBag(unittest.TestCase):

    def test_repr(self):
        """Test repr(Bag)."""
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)
        self.assertEqual(repr(bag), f"<Bag: '{bag.name}'>")

    def test_scope(self):
        """Test a Bag's scope."""
        # Get the Factors for the Sprinkler network
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        self.assertEqual(bag.scope, {'A', 'B', 'C', 'D', 'E'})

    def test_variable_elimination_single(self):
        """Test the variable elimination algorithm."""
        # Get the Factors for the Sprinkler network
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        # Compute the prior over C
        fC = bag.eliminate(['C'])

        self.assertAlmostEqual(fC['c0'], 0.48, places=2)
        self.assertAlmostEqual(fC['c1'], 0.52, places=2)
        self.assertAlmostEqual(fC.sum(), 1, places=8)

    def test_variable_elimination_multiple(self):
        """Test the variable elimination algorithm."""
        # Get the Factors for the Sprinkler network
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        # Compute the joint over A and C
        fAC = bag.eliminate(['A', 'C'])

        self.assertAlmostEqual(fAC['a0', 'c0'], 0.36, places=2)
        self.assertAlmostEqual(fAC['a0', 'c1'], 0.04, places=2)
        self.assertAlmostEqual(fAC['a1', 'c0'], 0.12, places=2)
        self.assertAlmostEqual(fAC['a1', 'c1'], 0.48, places=2)
        self.assertAlmostEqual(fAC.sum(), 1, places=8)

    def test_variable_elimination_with_evidence(self):
        """Test the variable elimination algorithm."""
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        # Compute the (unnormalized) factor over C and A=a1
        fC_a1 = bag.eliminate(['C'], {'A': 'a1'})

        self.assertAlmostEqual(fC_a1['c0'], 0.12, places=2)
        self.assertAlmostEqual(fC_a1['c1'], 0.48, places=2)

    def test_compute_posterior(self):
        """Test the function Bag.compute_posterior()."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        I = bag.compute_posterior(['I'], {}, [], {})
        self.assertAlmostEqual(I['i0'], 0.70, places=2)
        self.assertAlmostEqual(I['i1'], 0.30, places=2)

        S_i1 = bag.compute_posterior(['S'], {}, [], {'I': 'i1'})
        self.assertAlmostEqual(S_i1['s0'], 0.20, places=2)
        self.assertAlmostEqual(S_i1['s1'], 0.80, places=2)

        S_i0 = bag.compute_posterior(['S'], {}, [], {'I': 'i0'})
        self.assertAlmostEqual(S_i0['s0'], 0.95, places=2)
        self.assertAlmostEqual(S_i0['s1'], 0.05, places=2)

        G_I = bag.compute_posterior(['G'], {}, ['I'], {})
        self.assertAlmostEqual(G_I['i0', 'g1'], 0.20, places=2)
        self.assertAlmostEqual(G_I['i0', 'g2'], 0.34, places=2)
        self.assertAlmostEqual(G_I['i0', 'g3'], 0.46, places=2)

        s0_i0 = bag.compute_posterior([], {'S': 's0'}, [], {'I':'i0'})
        s1_i0 = bag.compute_posterior([], {'S': 's1'}, [], {'I':'i0'})
        self.assertAlmostEqual(s0_i0, 0.95, places=2)
        self.assertAlmostEqual(s1_i0, 0.05, places=2)

        s0G_i0 = bag.compute_posterior(['G'], {'S': 's0'}, [], {'I':'i0'})
        self.assertEqual(len(s0G_i0), 3)

    @unittest.skip('Not yet')
    def test_MAP(self):
        """Test the Bag.MAP() function."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        argmax_I = bag.MAP(['I'], {}, False)
        self.assertEqual(argmax_I, 'i0')

        argmax_G = bag.MAP(['G'], {}, False)
        self.assertEqual(argmax_G, 'g1')

        argmax_G = bag.MAP(['G'], {}, True)
        self.assertEqual(argmax_G[0], 'g1')
        self.assertAlmostEqual(argmax_G[1], 0.362)

    def test_P(self):
        """Test the function Bag.P()."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        I = bag.P('I')
        self.assertAlmostEqual(I['i0'], 0.7, places=3)
        self.assertAlmostEqual(I['i1'], 0.3, places=3)

        G_I = bag.P('G|I')
        self.assertEqual(G_I.scope, ['I', 'G'])
        self.assertEqual(G_I.conditioned, ['G'])
        self.assertEqual(G_I.conditioning, ['I'])

        I_g1 = bag.P('I|G=g1')
        self.assertAlmostEqual(I_g1['i0'], 0.387, places=3)
        self.assertAlmostEqual(I_g1['i1'], 0.613, places=3)

    def test_JPT(self):
        """Test Joint Probability Table."""
        factors = examples.get_student_CPTs()
        bag = Bag('Student', list(factors.values()))

        jpt = bag.eliminate(list(bag.scope)).normalize()
        self.assertAlmostEqual(jpt.sum(), 1, places=5)

    def test_as_dict(self):
        """Test serialization."""
        factors = examples.get_sprinkler_factors()
        bag = Bag('Sprinkler', factors)

        bd = bag.as_dict()

        self.assertEqual(bd['type'], 'Bag')
        self.assertEqual(bd['name'], bag.name)
        self.assertEqual(len(bd['factors']), len(bag))
