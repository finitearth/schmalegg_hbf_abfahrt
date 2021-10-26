import generate_random

if __name__ == '__main__':
    for _ in range(1000):
        routes = generate_random.generate_random_routes()

        for route1 in routes:
            c = 0
            for route2 in routes:
                if route1 == route2:
                    c += 1
                if c == 2:
                    print("ALLLLLLLLLLAAAAAAAAAAAARM")

    for route in routes:
        print(route)
