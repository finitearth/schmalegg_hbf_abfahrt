class Test:
    def __eq__(self, other):
        return True

a = Test()
b = Test()

c = [a, b]
print(a in c)