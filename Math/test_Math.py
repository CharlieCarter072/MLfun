from unittest import TestCase
from MathFuncts import *


class TestDotProd(TestCase):
    def test_dot_product(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)


class TestMatrixMultiply(TestCase):

    def test_mat_mul(self):
        self.assertEqual(mat_mul([[1,2,3],[4,5,6]], [[10,11],[20,21],[30,31]]), [[140, 146], [320, 335]])


