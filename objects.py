class Station:
    def __init__(self, x, y, capacity, name="1"):
        self.x = x
        self.y = y
        self.name = name
        self.capacity = capacity

    def __str__(self):
        return self.name

class Route:
    def __init__(self, station1, station2):
        self.station1 = station1
        self.station2 = station2

    def __eq__(self, other):
        if self.station1 == other.station1 and self.station2 == other.station2: return True
        elif self.station1 == other.station2 and self.station2 == other.station1: return True
        else: return False

    def __str__(self):
        return f"{self.station1} <-> {self.station2}"
