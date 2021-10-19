import random
import objects


def generate_random_routes(width=5, height=5, max_capacity=10, number_stations=10):
    stations = []
    for i, _ in enumerate(range(number_stations)):
        x = random.random()*width
        y = random.random()*height
        capacity = random.random()*max_capacity

        station = objects.Station(x, y, capacity, name=str(i))
        stations.append(station)

    routes = []
    for station1 in stations:
        for station2 in stations:
            if station1 == station2: continue
            if random.random() < 0.3:
                route = objects.Route(station1, station2)
                if route in routes: continue
                routes.append(route)

    return routes




