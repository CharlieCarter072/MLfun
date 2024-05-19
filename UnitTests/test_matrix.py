from unittest import TestCase
from Programs.matrix import *


class TestMatrix(TestCase):
    test_matrix = Matrix([[1, 2, 3], [4, 5, 6]])

    def test_edit_column(self):
        self.test_matrix.edit_column(1, Matrix([[7], [8]]))
        self.assertEqual(self.test_matrix.items, [[1, 7, 3], [4, 8, 6]])
        print(self.test_matrix.items)


class TestDotProd(TestCase):
    def test_dot_product(self):
        self.assertEqual(dot_product(Matrix([1, 2, 3]), Matrix([4, 5, 6])), 32)


class TestMatrixMultiply(TestCase):

    def test_mat_mul(self):
        mat1 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        mat2 = Matrix([[10, 11], [20, 21], [30, 31]])
        self.assertEqual(mat_mul(mat1, mat2).items, [[140, 146], [320, 335], [500, 524]])