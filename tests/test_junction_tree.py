# -*- coding: utf-8 -*-
import os
import unittest
import doctest
import logging
import itertools
import json

from tempfile import gettempdir

import pandas as pd

import thomas.core
from thomas.core import error
from thomas.core.cpt import CPT
from thomas.core.bayesiannetwork import BayesianNetwork, DiscreteNetworkNode
from thomas.core import examples


log = logging.getLogger(__name__)


class TestJunctionTree(unittest.TestCase):

    def setUp(self):
        self.Gs = examples.get_student_network()

    def test_ensure_cluster(self):
        """Test tree.ensure_cluster()."""
        jt = self.Gs.jt
        Q1 = {'D', 'G', 'I'}
        Q2 = {'L', 'G', 'S'}
        self.assertTrue(jt.get_node_for_set(Q1) is not None)
        self.assertTrue(jt.get_node_for_set(Q2) is None)

        jt.ensure_cluster(Q2)
        self.assertTrue(jt.get_node_for_set(Q2) is not None)


    def test_set_evidence_hard(self):
        """Test tree.set_evidence_hard()."""
        with self.assertRaises(error.InvalidStateError):
            self.Gs.jt.set_evidence_hard(I='i2')