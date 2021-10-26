import random
import objects


def generate_random_routes(width=5, height=5, max_capacity=10, number_stations=10):
    edge_proneness = random.random()

    stations = []
    for i, _ in enumerate(range(number_stations)):
        x = random.random() * width
        y = random.random() * height
        capacity = random.random() * max_capacity

        station = objects.Station(x, y, capacity, name=str(i))
        stations.append(station)

    routes = []
    for station1 in stations:
        for station2 in stations:
            if station1 == station2: continue
            if random.random() < edge_proneness:
                route = objects.Route(station1, station2)
                if not route in routes: routes.append(route)

    return routes


def generate_random_env():
    # erstmal immer die gleiche
    stations = [
        objects.Station(0, 0, 1),
        objects.Station(1, 1, 1),
        objects.Station(0, 1, 1),
        objects.Station(1, 0, 1),
        objects.Station(0.5, 0.5, 1),
    ]

    passengergroups = [objects.PassengerGroup(stations[0], stations[2], 1, None)]

    trains = None  # TODO implement trains

    return stations, passengergroups, trains
