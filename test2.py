class A:
    __slots__ = ["index", "value"]

    def __init__(self, index, value):
        self.index = index
        self.value = value

    def __repr__(self):
        return "{0}, {1}".format(self.index, self.value)


a = A(2, 3)
b = A(3, 4)
c = []
c.append((a, 1))
c.append((b, 2))
a_ = c[0][0]
a_.value = 1
print(a)

