class Matrix:
    def __init__(self, items=[]):
        self.items = items

    def __str__(self):
        return str(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def row_count(self):
        return len(self.items)

    def column_count(self):
        if self.row_count() == 0:
            return 0
        else:
            return len(self.items[0])

    def row(self, row_index):
        return self.items[row_index]

    def column(self, column_index):
        return [i[column_index] for i in self.items]

    def add_row(self, new_row):  # mostly useful for vectors, hopefully won't have to add an add_column method
        self.items.append(new_row)

