# -*- coding: utf-8 -*-
import unittest
import logging

from thomas.core.models.base import ProbabilisticModel

log = logging.getLogger(__name__)

class TestProbabilisticModel(unittest.TestCase):

    def test_compute_posterior(self):

        model = ProbabilisticModel()

        with self.assertRaises(NotImplementedError):
            model.compute_posterior([], {}, [], {})
