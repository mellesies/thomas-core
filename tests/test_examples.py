import unittest
import logging

from thomas.core.models.bn import BayesianNetwork
from thomas.core import examples

log = logging.getLogger(__name__)


class TestExamples(unittest.TestCase):

    def test_sprinkler_network(self):
        """Test loading the sprinkler network."""
        bn = examples.get_sprinkler_network()
        self.assertEqual(bn.name, 'Sprinkler')

    def test_student_network(self):
        """Test loading the sprinkler network."""
        bn = examples.get_student_network()
        self.assertEqual(bn.name, 'Student')

    def test_example7_network(self):
        """Test loading the sprinkler network."""
        bn = examples.get_example7_network()
        self.assertEqual(bn.vars, {'A', 'B', 'C'})

    def test_lungcancer_network(self):
        """Test loading the Lungcancer network."""
        bn = examples.get_lungcancer_network()

        self.assertTrue(isinstance(bn, BayesianNetwork))
        self.assertEqual(len(bn.nodes), 10)
