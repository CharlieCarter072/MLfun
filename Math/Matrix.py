class Matrix(list):  # A = Matrix([[1, 2],[3, 4]])     A @ B
    def __matmul__(self, B):
        A = self
        return Matrix([[sum(A[i][k] * B[k][j] for k in range(len(B))) for j in range(len(B[0])) ] for i in range(len(A))])