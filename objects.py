import numpy as np


class Station:
    def __init__(self, x, y, capacity, name=-1):
        self.x = x
        self.y = y
        self.name = name
        self.capacity = capacity
        self.passengers = [0, 0, 0, 0, 0]

    def getencoding(self):
        l = [0] * 5
        l[self.name] = 1

        return np.hstack([l, self.passengers]).astype(np.float32)

    def __str__(self):
        return self.name


class Route:
    def __init__(self, station1, station2):
        self.station1 = station1
        self.station2 = station2
        self.distance = ((station1.x - station2.x) ** 2 + (station1.y - station2.y)) ** (1 / 2)

    def __eq__(self, other):
        if self.station1 == other.station1 and self.station2 == other.station2:
            return True
        elif self.station1 == other.station2 and self.station2 == other.station1:
            return True
        else:
            return False

    def __str__(self):
        return f"{self.station1} <-> {self.station2}"


class PassengerGroup:
    def __init__(self, current_station, destination, n_people, target_time):
        self.current_station = current_station
        self.destination = destination
        self.n_people = n_people
        self.target_time = target_time

    def reached_destination(self):
        return self.destination == self.current_station


class Train:
    def __init__(self, capacity):
        self.speed = 0
        self.station = None
        self.capacity = capacity
        self.passengers = []
