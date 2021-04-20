# -*- coding: utf-8 -*-
import unittest
import logging

import thomas.core
from thomas.core.bayesiannetwork import BayesianNetwork
from thomas.core.reader import oobn

log = logging.getLogger(__name__)

class TestOOBNReader(unittest.TestCase):

    def setUp(self):
        self.maxDiff = None
        self.places = 3

    def test_oobn_reader(self):
        filename = thomas.core.get_pkg_data('prostatecancer.oobn')
        bn = oobn.read(filename)

        self.assertTrue(isinstance(bn, BayesianNetwork))

        grade = bn['grade'].cpt
        self.assertAlmostEqual(grade['g2'], 0.0185338)
        self.assertAlmostEqual(grade['g3'], 0.981466)

        cT = bn['cT'].cpt.reorder_scope(['grade', 'cT'])
        self.assertAlmostEqual(cT['g2', 'T2'], 0.0)
        self.assertAlmostEqual(cT['g2', 'T3'], 0.0)
        self.assertAlmostEqual(cT['g2', 'T4'], 1.0)
        self.assertAlmostEqual(cT['g3', 'T2'], 0.521457)
        self.assertAlmostEqual(cT['g3', 'T3'], 0.442157)
        self.assertAlmostEqual(cT['g3', 'T4'], 0.0363858)

        cN = bn['cN'].cpt.reorder_scope(['edition', 'cT'])
        self.assertAlmostEqual(cN['TNM 6', 'T2', 'NX'], 0.284264)
        self.assertAlmostEqual(cN['TNM 6', 'T2', 'N0'], 0.680203)
        self.assertAlmostEqual(cN['TNM 6', 'T2', 'N1'], 0.035533)

        cTNM = bn['cTNM'].cpt.reorder_scope(['cN', 'cT', 'edition', 'cTNM'])
        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 6', 'I'], 0.0)
        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 6', 'II'], 1.0)
        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 6', 'III'], 0.0)
        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 6', 'IV'], 0.0)

        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 7', 'I'], 0.522727)
        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 7', 'II'], 0.454545)
        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 7', 'III'], 0.0)
        self.assertAlmostEqual(cTNM['NX', 'T2', 'TNM 7', 'IV'], 0.0227273)
