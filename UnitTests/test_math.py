from unittest import TestCase


class TestDotProd(TestCase):
    def test_dot_product(self):
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)


class TestMatrixMultiply(TestCase):

    def test_mat_mul(self):
        mat1 = Matrix([[1,2,3],[4,5,6],[7,8,9]])
        mat2 = Matrix([[10,11],[20,21],[30,31]])
        #self.assertEqual(mat_mul(mat1, mat2), [[140, 146], [320, 335]])
        print(mat_mul(mat1, mat2))  # [[140, 146], [320, 335], [500, 524]


