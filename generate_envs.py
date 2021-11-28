import random
import objects
from itertools import product


def generate_random_routes(max_capacity=10, max_n_stations=20):
    def _is_reachable(station1_, station2_, i_max, visited=None, i=0):
        if visited is None:
            visited = []

        if station1 in visited: return False
        visited.append(station1_)
        if station1 in station2_.reachable_stops: return True

        for station in station1_.reachable_stops:
            if i > i_max: return False
            if _is_reachable(station, station2_, i_max, visited=visited, i=i + 1):
                return True
        return False
    n_stations = random.randint(3, max_n_stations)
    edge_proneness = random.random() / n_stations ** 2

    routes = objects.Routes()

    stations = [objects.Station(random.random() * max_capacity, name=i) for i in range(n_stations)]

    middle_index = len(stations) // 2
    # create random connections
    for station1, station2 in product(stations[middle_index:], stations[:middle_index]):
        if random.random() < edge_proneness:
            routes.add(station1, station2)

    # check if every station is reachable from another station; if not create a new connection
    station_connections = list(product(stations[middle_index:], stations[:middle_index]))
    for station1, station2 in random.sample(station_connections, len(station_connections)):
        if not _is_reachable(station1, station2, n_stations ** 2):
            routes.add(station1, station2)

    return routes.get_all_routes(), stations


def generate_evaluation_env():
    raise NotImplementedError


def generate_random_env(n_passenger_group_max=5, n_trains_max=5):
    routes, stations = generate_random_routes()
    n_passenger_group = max(1, int(n_passenger_group_max * random.random()))

    for _ in range(n_passenger_group):
        destination = random.choice(stations)
        target_time = random.randint(3, 30)
        n_passenger = random.randint(1, 10)
        passenger_group = objects.PassengerGroup(destination, n_passenger, target_time)
        station = destination
        while destination == station:
            station = random.choice(stations)
        station.passengers.append(passenger_group)

    trains = []
    n_trains = max(1, int(n_trains_max * random.random()))
    for _ in range(n_trains):  # TODO wildcard trains
        station = random.choice(stations)
        capacity = 100# random.randint(1, 10)
        train = objects.Train(station, capacity)
        trains.append(train)

    return routes, stations, trains


def generate_example_enviroment():
    stations = [
        objects.Station(12, name=0),
        objects.Station(1, name=1),
        objects.Station(13, name=2),
        objects.Station(1, name=3),
        objects.Station(2, name=4),
    ]
    routes = objects.Routes()
    routes.add(stations[0], stations[1])
    routes.add(stations[1], stations[2])
    routes.add(stations[1], stations[3])
    routes.add(stations[3], stations[4])
    # for station1 in stations:
    #     for station2 in stations:
    #         routes.add(station1, station2)

    stations[1].passengers = [objects.PassengerGroup(stations[0], 21, 0)]
    # stations[2].passengers = [objects.PassengerGroup(stations[1], 21, 0)]
    # stations[4].passengers = [objects.PassengerGroup(stations[3], 21, 0)]

    trains = [objects.Train(stations[3], 4)]

    return routes.get_all_routes(), stations, trains
