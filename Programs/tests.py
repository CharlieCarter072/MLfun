from math_functions import *

a = Matrix([[1, 2, 3]])
b = Matrix([[1], [2], [3]])

print(b.size())
print(type(b[:-1:]))
print(Matrix(b[:-1:]))
