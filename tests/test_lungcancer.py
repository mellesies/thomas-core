# -*- coding: utf-8 -*-
import os
import unittest
import doctest
import logging
import itertools
import random

import thomas.core
from thomas.core.bayesiannetwork import BayesianNetwork
from thomas.core import examples
from thomas.core.reader import oobn

log = logging.getLogger(__name__)

# @unittest.skip
class TestLungCancerNetwork(unittest.TestCase):

    def setUp(self):
        pkg_path = os.path.dirname(thomas.core.__file__)
        fullpath = os.path.join(pkg_path, 'data', 'lungcancer.json')

        self.lungcancer = BayesianNetwork.open(fullpath)
        self.maxDiff = None
        self.places = 3

    def get_OOBN_path(self):
        """Return the path to the package's lungcancer.oobn"""
        pkg_path = os.path.dirname(thomas.core.__file__)
        fullpath = os.path.join(pkg_path, 'data', 'lungcancer.oobn')
        return fullpath

    @unittest.skip
    def test_elimination_order_importance(self):
        """This test may take a very long time and may in fact be intractable.
        """
        self.lungcancer.reset_evidence()

        nodes = list(self.lungcancer.nodes.keys())
        orders = list(itertools.permutations(nodes))

        for order in random.sample(orders, 5):
            print()
            print(order)
            self.lungcancer._jt = None
            self.lungcancer.elimination_order = order

            priors = self.lungcancer.get_probabilities()
            cTNM = priors['cTNM']
            self.assertAlmostEquals(cTNM['1A'], 0.0966, places=self.places)
            self.assertAlmostEquals(cTNM['1B'], 0.1058, places=self.places)
            self.assertAlmostEquals(cTNM['2A'], 0.0172, places=self.places)


        self.lungcancer.elimination_order = None

    def test_node_T(self):
        # T = self.lungcancer.P('T')
        P = self.lungcancer.compute_marginals(['T'])
        T = P['T']
        self.assertAlmostEquals(T['1A'], 0.0965, places=self.places)
        self.assertAlmostEquals(T['1B'], 0.0819, places=self.places)
        self.assertAlmostEquals(T['2A'], 0.2051, places=self.places)
        self.assertAlmostEquals(T['2B'], 0.0755, places=self.places)
        self.assertAlmostEquals(T['3'],  0.1757, places=self.places)
        self.assertAlmostEquals(T['4'],  0.2940, places=self.places)
        self.assertAlmostEquals(T['X'],  0.0713, places=self.places)

    def test_node_cTNM(self):
        # cTNM = self.lungcancer.P('cTNM')
        # P = self.lungcancer.compute_marginals(['cTNM'])
        bn = oobn.read(self.get_OOBN_path())

        P = bn.compute_marginals(['cTNM'])
        cTNM = P['cTNM']
        self.assertAlmostEquals(cTNM['1A'], 0.0966, places=self.places)
        self.assertAlmostEquals(cTNM['1B'], 0.1058, places=self.places)
        self.assertAlmostEquals(cTNM['2A'], 0.0172, places=self.places)
        self.assertAlmostEquals(cTNM['2B'], 0.0400, places=self.places)
        self.assertAlmostEquals(cTNM['3A'], 0.1261, places=self.places)
        self.assertAlmostEquals(cTNM['3B'], 0.1467, places=self.places)
        self.assertAlmostEquals(cTNM['4'],  0.4462, places=self.places)
        self.assertAlmostEquals(cTNM['X'],  0.0214, places=self.places)

    def test_node_cTNM_TNM1A(self):
        # cTNM = self.lungcancer.P('cTNM|TNM=1A')
        # P = self.lungcancer.compute_marginals(['cTNM'], {'TNM': '1A'})
        bn = oobn.read(self.get_OOBN_path())

        P = bn.compute_marginals(['cTNM'], {'TNM': '1A'})
        cTNM = P['cTNM']
        self.assertAlmostEquals(cTNM['1A'], 0.8402, places=self.places)
        self.assertAlmostEquals(cTNM['1B'], 0.0153, places=self.places)
        self.assertAlmostEquals(cTNM['2A'], 0.0130, places=self.places)
        self.assertAlmostEquals(cTNM['2B'], 0.0156, places=self.places)
        self.assertAlmostEquals(cTNM['3A'], 0.0118, places=self.places)
        self.assertAlmostEquals(cTNM['3B'], 0.0401, places=self.places)
        self.assertAlmostEquals(cTNM['4'],  0.0151, places=self.places)
        self.assertAlmostEquals(cTNM['X'],  0.0490, places=self.places)

    def test_node_T_N0(self):
        # T = self.lungcancer.P('T|N=0')
        # P = self.lungcancer.compute_marginals(['T'], {'N': '0'})
        bn = oobn.read(self.get_OOBN_path())

        P = bn.compute_marginals(['T'], {'N': '0'})
        T = P['T']
        self.assertAlmostEquals(T['1A'], 0.1917, places=self.places)
        self.assertAlmostEquals(T['1B'], 0.1339, places=self.places)
        self.assertAlmostEquals(T['2A'], 0.2503, places=self.places)
        self.assertAlmostEquals(T['2B'], 0.0777, places=self.places)
        self.assertAlmostEquals(T['3'],  0.1511, places=self.places)
        self.assertAlmostEquals(T['4'],  0.1409, places=self.places)
        self.assertAlmostEquals(T['X'],  0.0544, places=self.places)

    def test_node_cT_T1A_TNM1A(self):
        # cT = self.lungcancer.P('cT|T=1A,TNM=1A')
        # P = self.lungcancer.compute_marginals(['cT'], {'T': '1A', 'TNM': '1A'})
        bn = oobn.read(self.get_OOBN_path())

        P = bn.compute_marginals(['cT'], {'T': '1A', 'TNM': '1A'})
        cT = P['cT']
        self.assertAlmostEquals(cT['1'],  0.5589, places=self.places)
        self.assertAlmostEquals(cT['1A'], 0.3195, places=self.places)
        self.assertAlmostEquals(cT['1B'], 0.0031, places=self.places)
        self.assertAlmostEquals(cT['2'],  0.0150, places=self.places)
        self.assertAlmostEquals(cT['2A'], 0.0012, places=self.places)
        self.assertAlmostEquals(cT['2B'], 0.0009, places=self.places)
        self.assertAlmostEquals(cT['3'],  0.0178, places=self.places)
        self.assertAlmostEquals(cT['4'],  0.0317, places=self.places)
        self.assertAlmostEquals(cT['X'],  0.0520, places=self.places)

