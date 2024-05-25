from Programs.math_functions import *


class Matrix:
    def __init__(self, items=[]):
        self.items = items

    def __str__(self):
        return str(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def row_count(self):
        return len(list(self.items))

    def column_count(self):
        if self.row_count() == 0:
            return 0
        else:
            return len(list(self.items[0]))

    def row(self, row_index):
        return Matrix(self.items[row_index])

    def column(self, column_index):
        return Matrix([i[column_index] for i in self.items])

    def add_row(self, new_row):  # mostly useful for vectors, hopefully won't have to add an add_column method
        self.items.append(new_row)

    def get_label(self):
        return Matrix([self.items[0]])

    def get_data(self):
        return Matrix(self.items[1::])

    def edit_row(self, row_index, new_row):
        self.items[row_index] = new_row

    def edit_column(self, column_index, new_column):
        for i in range(self.row_count()):
            self.items[i][column_index] = new_column[i][0]

    def transpose(self):
        return Matrix([self.column(i).items for i in range(self.column_count())])


def mat_mul(a, b):
    final_matrix = Matrix(
        [[dot_product(a.row(i), b.column(j)) for j in range(b.column_count())] for i in range(a.row_count())]
    )
    return final_matrix


def vectorize(lst):
    return Matrix([[i] for i in lst])


def dot_product(a, b):  # dot product, scores similarity
    total = 0
    for i in range(a.row_count()):
        total += a[i] * b[i]
    return total
