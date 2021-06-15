# -*- coding: utf-8 -*-
import unittest
import logging

import numpy as np

from thomas.core.factors import Factor
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
            S_I._repr_html_()
        except Exception:
            self.fail('Creating an HTML representation raised an exception?')

    def test_mul_cpt(self):
        """Test CPT multiplication with another Factor/CPT."""
        CPTs = examples.get_student_CPTs()
        I = CPTs['I']
        S_I = CPTs['S']

        # Multiplying a CPT should yield a Factor
        SI = S_I.mul(I)
        self.assertIsInstance(SI, Factor)

    def test_mul_scalar(self):
        """Test CPT multiplication with a scalar."""
        CPTs = examples.get_student_CPTs()
        I = CPTs['I']
        S_I = CPTs['S']

        # Multiplying a CPT should yield a Factor
        Sx3 = S_I.mul(3)
        self.assertIsInstance(Sx3, Factor)

    def test_div_cpt(self):
        """Test CPT division."""
        CPTs = examples.get_student_CPTs()
        I = CPTs['I']
        S_I = CPTs['S']

        # Dividing a CPT should yield a Factor
        SI = S_I.div(I)
        self.assertIsInstance(SI, Factor)

    def test_mul_div(self):
        """Test division after multiplication for equality."""
        CPTs = examples.get_student_CPTs()
        I = CPTs['I']
        S_I = CPTs['S']

        # Multiplying a CPT should yield a Factor
        result = S_I.mul(I).div(I)
        self.assertIsInstance(result, Factor)
        self.assertTrue(np.allclose(result.values, S_I.values))

    # @unittest.skip('no more ....')
    # def test_unstack(self):
    #     """Test CPT.unstack()."""
    #     CPTs = examples.get_student_CPTs()
    #     G_ID = CPTs['G']
    #
    #     self.assertIsInstance(G_ID.unstack(), pd.DataFrame)
    #
    #     conditioning = list(G_ID.unstack().index.names)
    #     self.assertEqual(conditioning, ['I', 'D'])
    #
    #     rowidx = list(G_ID.unstack('D').index.names)
    #     colname = G_ID.unstack('D').columns.name
    #     self.assertTrue('D' not in rowidx)
    #     self.assertEqual(colname, 'D')

    def test_as_factor(self):
        """Test CPT.as_factor()."""
        CPTs = examples.get_student_CPTs()
        G_ID = CPTs['G']
        self.assertIsInstance(G_ID.as_factor(), Factor)
