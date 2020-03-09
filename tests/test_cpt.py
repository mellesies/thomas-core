# -*- coding: utf-8 -*-
import unittest
import doctest
import logging

import json

import pandas as pd

import thomas.core
from thomas.core.factor import Factor
from thomas.core.cpt import CPT
from thomas.core.jpt import JPT
from thomas.core import examples

log = logging.getLogger(__name__)

class TestCPT(unittest.TestCase):

    def test_repr(self):
        """Test repr(jpt)."""
        CPTs = examples.get_student_CPTs()
        I = CPTs['I']
        S_I = CPTs['S']

        self.assertEqual(I.display_name, 'P(I)')
        self.assertEqual(S_I.display_name, 'P(S|I)')

        # Unfortunately, pandas.DataFrame._repr_html_() returns syntactically
        # invalid html, so we cannot validate the html itself :-(.
        try:
            html = S_I._repr_html_()
        except Exception as e:
            self.fail('Creating an HTML representation raised an exception?')

    def test_unstack(self):
        """Test CPT.unstack()."""
        CPTs = examples.get_student_CPTs()
        G_ID = CPTs['G']

        self.assertIsInstance(G_ID.unstack(), pd.DataFrame)

        conditioning = list(G_ID.unstack().index.names)
        self.assertEqual(conditioning, ['I', 'D'])

        rowidx = list(G_ID.unstack('D').index.names)
        colname = G_ID.unstack('D').columns.name
        self.assertTrue('D' not in rowidx)
        self.assertEqual(colname, 'D')


    def test_as_factor(self):
        """Test CPT.as_factor()."""
        CPTs = examples.get_student_CPTs()
        G_ID = CPTs['G']
        self.assertIsInstance(G_ID.as_factor(), Factor)


