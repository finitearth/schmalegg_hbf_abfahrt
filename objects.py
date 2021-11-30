import numpy as np

NODE_FEATURES = 4


class Station:
    def __init__(self, capacity, name=-1):
        self.name = name
        self.capacity = capacity
        self.passengers = []
        self.reachable_stops = []
        self.vector = None
        self.input_vector = np.ones(NODE_FEATURES) * 0.95 + np.random.rand(NODE_FEATURES) * 0.1  # Values from 0.95 to 1.05

    def getencoding(self):
        return self.input_vector
        # passenger_encoding = np.zeros(shape=(5,))
        # for p in self.passengers:
        #     passenger_encoding[int(p.destination)] += 1
        #
        # return passenger_encoding

    def __int__(self):
        return self.name

    def __repr__(self):
        return f"Station {int(self)}"


class Routes:
    def __init__(self):
        self.station1s = []
        self.station2s = []

    def add(self, station1, station2):
        assert isinstance(station1, Station), f"{station1} is not of type Station"
        assert isinstance(station2, Station), f"{station2} is not of type Station"
        if station1 == station2 or station1 in station2.reachable_stops: return

        station1.reachable_stops.append(station2)
        station2.reachable_stops.append(station1)
        self.station1s.append(station1)
        self.station2s.append(station2)

    def get_all_routes(self):
        s1 = [int(s) for s in self.station1s]
        s2 = [int(s) for s in self.station2s]
        return s1 + s2, s2 + s1


class PassengerGroup:
    def __init__(self, destination, n_people, target_time):
        self.destination = destination
        self.n_people = n_people
        self.target_time = target_time

    def reached_destination(self, current_station):
        assert isinstance(current_station, Station), f"{current_station} is not of type Station"
        # print(f"{current_station} == {self.destination}")
        return self.destination == current_station


class Train:
    def __init__(self, station, capacity):
        self.speed = 1
        self.station = station
        self.destination = None
        self.capacity = capacity
        self.passengers = []

    def reached_next_stop(self):
        if self.destination is not None:
            self.station = self.destination
        return True  # TODO

    def reroute_to(self, destination):
        assert isinstance(destination, Station), f"{destination} is not of type Station"
        self.destination = destination

    def onboard(self, passenger_group):
        assert isinstance(passenger_group, PassengerGroup), f"{passenger_group} is not of type PassengerGroup"
        self.passengers.append(passenger_group)
        self.station.passengers.remove(passenger_group)

    def deboard(self, passenger_group):
        assert isinstance(passenger_group, PassengerGroup), f"{passenger_group} is not of type PassengerGroup"
        self.passengers.remove(passenger_group)
        if not passenger_group.reached_destination(self.station):
            passenger_group.current_station = self.station
            self.station.passengers.append(passenger_group)
