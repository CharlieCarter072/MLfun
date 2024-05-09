class Matrix:
    def __init__(self, items=[]):
        self.items = items

    def __str__(self):
        return str(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def rows(self):
        return len(self.items)

    def columns(self):
        if self.rows() == 0:
            return 0
        else:
            return len(self.items[0])


test = Matrix([[1, 2], [3, 4]])


