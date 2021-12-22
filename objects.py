import random

import networkx as nx
import numpy as np
import json

from matplotlib import pyplot as plt
from networkx import fast_gnp_random_graph


class EnvBlueprint:
    def __init__(self, stations=None, routes=None, passengers=None, trains=None, name=None):
        self.stations = stations
        self.routes = routes
        self.passengers = [] if passengers is None else passengers
        self.trains = trains
        self.name = name

    def read(self, file):
        with open(file, "r") as f:
            env_dict = json.load(f)

        routes = Routes()

        stations_dict = {}
        for station in env_dict["stations"]:
            stations_dict[str(station["name"])] = Station(station["capacity"], station["name"])
        stations_list = [s for s in stations_dict.values()]

        for route in env_dict["routes"]:
            routes.add(
                stations_dict[str(route[0])], stations_dict[str(route[1])]
            )

        passengers = []
        for passenger in env_dict["passengers"]:
            pg = PassengerGroup(stations_dict[str(passenger["destination"])],
                                passenger["n_people"],
                                passenger["target_time"])
            passengers.append(pg)
            station = stations_dict[str(passenger["start_station"])]
            station.passengers.append(pg)

        trains = []
        for train in env_dict["trains"]:
            trains.append(Train(stations_dict[str(train["station"])], train["capacity"]))

        # self.name = env_dict["name"]
        self.passengers = passengers
        self.routes = routes.get_all_routes()
        self.trains = trains
        self.stations = stations_list

    def save(self):
        text = {
            "name": self.name,
            "stations": [{"name": station.name, "capacity": station.capacity} for station in self.stations],
            "routes": [route for route in self.routes],
            "passengers": [{"destination": str(passenger.destination),
                            "n_people": passenger.n_people,
                            "target_time": passenger.target_time,
                            "start_station": str(passenger.start_station)}
                           for passenger in self.passengers],
            "trains": [{"station": str(train.station), "capacity": train.capacity} for train in self.trains]}

        with open(self.name+".json", "w") as f:
            json.dump(text, f)

    def get(self):
        return self.routes, self.stations, self.trains

    def random(self, n_max_stations):
        n_station = int(max(7, n_max_stations*random.random()))

        p = 2 / (n_station + 1)
        graph = fast_gnp_random_graph(n_station, p)
        c = nx.k_edge_augmentation(graph, 1)
        graph.add_edges_from(c)

        edges = list(graph.edges)

        stations = []
        for i in range(n_station):
            capacity = random.randint(5, 100)
            stations.append(Station(capacity=capacity, name=i))
        self.stations = stations
        routes = Routes()
        for e in edges:
            routes.add(stations[e[0]], stations[e[1]])
        self.routes = routes.get_all_routes()

        n_passenger_group_max = 3
        n_passenger_group = max(1, int(n_passenger_group_max * random.random()))
        for _ in range(n_passenger_group):
            station = random.choice(stations)
            target_time = random.randint(3, 30)
            n_passenger = random.randint(1, 10)

            destination = station
            while destination == station:
                station = random.choice(stations)
            passenger_group = PassengerGroup(destination, n_passenger, target_time)
            station.passengers.append(passenger_group)
            passenger_group.start_station = station
            self.passengers.append(passenger_group)

        n_trains_max = 1
        trains = []
        n_trains = max(1, int(n_trains_max * random.random()))
        for _ in range(n_trains):  # TODO wildcard trains
            station = random.choice(stations)
            capacity = 100  # random.randint(1, 10)
            train = Train(station, capacity)
            trains.append(train)
        self.trains = trains

    def render(self):
        edges = self.routes
        edges = list(zip(edges[0], edges[1]))
        graph = nx.Graph()
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
        nx.draw(graph)
        plt.show()


class Station:
    def __init__(self, capacity, name=-1, n_node_features=4):
        self.name = name
        self.capacity = capacity
        self.passengers = []
        self.reachable_stops = []
        self.vector = None
        self.input_vector = None

    def set_input_vector(self, n_node_features):
        self.input_vector = np.ones(n_node_features) * 0.95 + np.random.rand(n_node_features) * 0.1

    def getencoding(self):
        return self.input_vector

    def __int__(self):
        return int(self.name)

    def __repr__(self):
        return str(int(self))


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
        assert isinstance(destination, Station), f"{destination} is not of type Station"
        self.destination = destination
        self.n_people = n_people
        self.target_time = target_time
        self.start_station = None

    def reached_destination(self, current_station):
        assert isinstance(current_station, Station), f"{current_station} is not of type Station"
        return self.destination == current_station


class Train:
    def __init__(self, station, capacity):
        assert isinstance(station, Station), f"{station} is not of type Station"
        self.speed = 1
        self.station = station
        self.destination = None
        self.capacity = capacity
        self.passengers = []

    def reached_next_stop(self):
        if self.destination is not None:
            self.station = self.destination
        return True  # TODO

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

    def reroute_to(self, destination):
        assert isinstance(destination, Station), f"{destination} is not of type Station"
        self.destination = destination
