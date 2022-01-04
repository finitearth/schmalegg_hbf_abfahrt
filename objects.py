import random
import re

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

    def read_txt(self, file_path):
        with open(file_path, 'r') as f:
            text = f.read()
        text = "".join([t + "\n" for t in text.split("\n") if "#" not in t])
        splits = re.split(r"(\[Stations\]|\[Lines\]|\[Passengers\]|\[Trains\])", text)

        for i, split in enumerate(splits):
            if "[Stations]" in split:
                stations_text = splits[i + 1].replace("[Stations]\n", "")
            elif "[Lines]" in split:
                lines_text = splits[i + 1].replace("[Lines]\n", "")
            elif "[Passengers]" in split:
                passengers_text = splits[i + 1].replace("[Passengers]\n", "")
            elif "[Trains]" in split:
                trains_text = splits[i + 1].replace("[Trains]\n", "")

        single_stations = stations_text.split('\n')
        station_list = []
        stations_dict = {}
        for i, s in enumerate([s for s in single_stations if s != "" and s != " "]):
            ss = s.split(" ")
            sss = Station(int(ss[0][1]), i)
            station_list.append(sss)
            stations_dict[ss[0]] = sss

        routes = Routes()
        single_lines = lines_text.split('\n')
        for l in single_lines:
            if l == "" or l == " ": continue
            ll = l.split(" ")
            routes.add( stations_dict[ll[1]], stations_dict[ll[2]], stations_dict[ll[0]], stations_dict[ll][3], stations_dict[ll][4])

        single_trains = trains_text.split('\n')
        train_list = []
        for t in single_trains:
            if t == "" or t == " ": continue
            tt = t.split(" ")
            ttt = stations_dict[tt[1]] if tt[1] != "*" else list(stations_dict.values())[0]
            train_list.append(Train(ttt, int(tt[3]), name=int(tt[0][1:])))

        passengers_text = passengers_text.split('\n')
        passenger = []
        for p in passengers_text:
            if p == "" or p == " ": continue
            pp = p.split(" ")
            ppp = PassengerGroup(stations_dict[pp[2]], pp[3], pp[4])
            passenger.append(ppp)
            stations_dict[pp[1]].passengers.append(ppp)

        self.passengers = passenger
        self.routes = routes.get_all_routes()
        self.trains = train_list
        self.stations = station_list

    def read_json(self, file=None, text=None):
        if file is not None:
            with open(file, "r") as f:
                env_dict = json.load(f)
        else:
            env_dict = json.load(text)

        routes = Routes()

        stations_dict = {}
        for station in env_dict["stations"]:
            stations_dict[str(station["name"])] = Station(station["capacity"], station["name"])
        stations_list = [s for s in stations_dict.values()]

        for id, route in enumerate(env_dict["routes"]):
            max_capacity = 10 
            capacity =  max(1, int(max_capacity*random.random()))
            max_length = 15
            length = max_length*random.random()
            routes.add(
                stations_dict[str(route[0])], stations_dict[str(route[1])], id, capacity, length
                
            )

        passengers = []
        for passenger in env_dict["passengers"]:
            pg = PassengerGroup(stations_dict[str(passenger["destination"])],
                                passenger["n_people"],
                                passenger["target_time"],
                                stations_dict[str(passenger["start_station"])])
            passengers.append(pg)
            station = stations_dict[str(passenger["start_station"])]
            station.passengers.append(pg)

        trains = []
        for i, train in enumerate(env_dict["trains"]):
            trains.append(Train(stations_dict[str(train["station"])], train["capacity"], name=str(i), speed=1))

        # self.name = env_dict["name"]
        self.passengers = passengers
        self.routes = routes.get_all_routes()
        self.trains = trains
        self.stations = stations_list

    def save(self, save_to_disk=True):
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
        if save_to_disk:
            with open(self.name + ".json", "w") as f:
                json.dump(text, f)
        else: return text

    def get(self):
        stations = [Station(s.capacity, i) for i, s in enumerate(self.stations)]
        for sc, so in zip(stations, self.stations):
            sc.passengers = [PassengerGroup(stations[int(p.destination)], p.n_people, p.target_time, p.start_station)
                             for p in so.passengers]
            sc.reachable_stops = [stations[int(s)] for s in so.reachable_stops]
            sc.input_vector = so.input_vector

        trains = [Train(stations[int(t.station)], t.capacity, t.name, t.speed) for t in self.trains]
        # self.graph = nx.Graph()
        # for s1, s2 in zip(self.routes[0], self.routes[1]):
        #     self.graph.add_edge(s1, s2)
        return self.routes, stations, trains

    def random(self, n_max_stations):
        n_station = int(max(7, n_max_stations * random.random()))

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
        for i,e in enumerate(edges):
            max_capacity = 10 
            capacity =  max(1, int(max_capacity*random.random()))
            max_length = 15
            length = max_length*random.random()
            routes.add(stations[e[0]], stations[e[1]], i, capacity, length)
        self.routes = routes.get_all_routes()

        n_passenger_group_max = 1
        n_passenger_group = max(1, int(n_passenger_group_max * random.random()))
        for _ in range(n_passenger_group):
            station = random.choice(stations)
            target_time = random.randint(3, 30)
            n_passenger = random.randint(1, 10)

            destination = station
            while destination == station:
                station = random.choice(stations)
            passenger_group = PassengerGroup(destination, n_passenger, target_time, station)
            station.passengers.append(passenger_group)
            passenger_group.start_station = station
            self.passengers.append(passenger_group)

        n_trains_max = 1
        trains = []
        n_trains = max(1, int(n_trains_max * random.random()))
        for i in range(n_trains):  # TODO wildcard trains
            station = random.choice(stations)
            capacity = 100  # random.randint(1, 10)
            train = Train(station, capacity, name=str(i), speed=1)
            trains.append(train)
        self.trains = trains

    def render(self):
        edges = self.routes
        edges = list(zip(edges[0], edges[1]))
        graph = nx.Graph()
        for edge in edges:
            graph.add_edge(edge[0], edge[1])
        nx.draw(graph, with_labels=True)
        plt.show()


class Station:
    def __init__(self, capacity, name=-1):
        self.name = name
        self.capacity = capacity
        self.passengers = []
        self.reachable_stops = []
        self.vector = None
        self.input_vector = None

    def set_input_vector(self, config):
        self.input_vector = np.ones(config.n_node_features) * (1 - config.range_inputvec) \
                          + np.random.rand(config.n_node_features) * config.range_inputvec * 2

    def get_encoding(self):
        return self.input_vector

    def __int__(self):
        return int(self.name)

    def __eq__(self, other):
        return int(self) == int(other)

    # def __repr__(self):
    #     return str(int(self))


class Routes:
    def __init__(self):
        self.station1s = []
        self.station2s = []

    def add(self, station1, station2, id, capacity, length):
        self.id = id
        self.capacity = capacity
        self.length = length
        assert isinstance(station1, Station), f"{station1} is not of type Station"
        assert isinstance(station2, Station), f"{station2} is not of type Station"
        if station1 == station2 or station1 in station2.reachable_stops: return

        station1.reachable_stops.append(station2)
        station2.reachable_stops.append(station1)
        self.station1s.append(station1)
        self.station2s.append(station2)

    def get_all_routes(self):
        s1 = [int(s) for s in self.station1s+self.station2s]
        s2 = [int(s) for s in self.station2s+self.station1s]
        return s1, s2


class PassengerGroup:
    def __init__(self, destination, n_people, target_time, start_station):
        assert isinstance(destination, Station), f"{destination} is not of type Station"
        self.destination = destination
        self.n_people = n_people
        self.target_time = target_time
        self.start_station = start_station

    def reached_destination(self, current_station):
        assert isinstance(current_station, Station), f"{current_station} is not of type Station"
        return self.destination == current_station


class Train:
    def __init__(self, station, capacity, name, speed):
        assert isinstance(station, Station), f"{station} is not of type Station"
        self.speed = 1 #speed 
        self.station = station
        self.destination = None
        self.capacity = capacity
        self.passengers = []
        self.name = name
        self.input_vector = None

    def set_input_vector(self, config):
        self.input_vector = np.ones(config.n_node_features) * (1 - config.range_inputvec) \
                        + np.random.rand(config.n_node_features) * config.range_inputvec * 2

    def get_encoding(self):
        return self.input_vector

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
        # else: print("reached :) ")

    def reroute_to(self, destination):
        assert isinstance(destination, Station), f"{destination} is not of type Station"
        self.destination = destination

    def __int__(self):
        return int(self.name)
