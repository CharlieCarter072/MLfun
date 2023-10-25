from unittest import TestCase
from MathFuncts import *


class TestDotProd(TestCase):
    def test_dot_product(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)




