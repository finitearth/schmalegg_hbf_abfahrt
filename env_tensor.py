import torch


class TensorEnv:

    def __init__(self):
        pass

    def step(self, actions):
        pass

    def reset(self):
        pass

    def generate_random(self):
        pass

    def read_from_txt(self, path):
        pass


# 2 example enviroments in one batch
# env 1: one train; three stations; 1 passenger
# env 2: two trains, 2 stations, 2 passengers

def close_or_greater(a, b):
    close = torch.isclose(a, b)
    greater = torch.greater(a, b)

    return torch.logical_or(close, greater)


def greater_not_close(a, b):
    close = torch.isclose(a, b)
    greater = torch.greater(a, b)

    return torch.logical_and(greater, torch.logical_not(close))


def less_not_close(a, b):
    close = torch.isclose(a, b)
    less = torch.less(a, b)

    return torch.logical_and(less, torch.logical_not(close))


def close_or_less(a, b):
    close = torch.isclose(a, b)
    greater = torch.less(a, b)

    return torch.logical_or(close, greater)


def get_station_adj_routes():
    env1_adj_tensor = torch.Tensor([
        [1, 1, 0], #station1
        [0, 1, 0], #station2
        [0, 1, 1], #station3
        [0, 0, 0]  #train1

    ])
    length_routes1 = torch.Tensor([
        [1, 3, 2],
        [2, 1, 2],
        [0, 4, 1]
    ])
    env2_adj_tensor = torch.Tensor([
        [1, 1],
        [1, 1]
    ])
    length_routes2 = torch.Tensor([

    ])
    return env1_adj_tensor, length_routes1, env2_adj_tensor, length_routes2


def get_train_tensor():
    env1_train_tensor = torch.Tensor([
        [
            [float("NaN"),  float("NaN"), 0],
            [float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN")]
        ]
    ])

    env1_trains_velocity = torch.Tensor([0.1])

    env2_train_tensor = torch.Tensor([
        [
            [0, .3],
            [0, 0]
        ],
        [
            [0, 0],
            [.3, 0]
        ]
    ])

    env2_trains_velocity = torch.Tensor([0.1, 0.15])

    return env1_train_tensor, env1_trains_velocity, env2_train_tensor, env2_trains_velocity


def get_passenger_tensor():
    env1_delay_tensor = torch.Tensor([[
        [float('nan'), float('nan'), float('nan')], # Spalten: Ziel
        [float('nan'), float('nan'), float('nan')], # Zeile: aktueller Bahnhof
        [float('nan'),  -12, float('nan')],
        [float("nan"), float("nan"), float("nan")]
    ]])

    env2_delay_tensor = torch.Tensor([
        [
            [float('nan'), -3],
            [float('nan'), float('nan')]
        ],
        [
            [float('nan'), float('nan')],
            [5, float('nan')]
        ]
    ])
    return env1_delay_tensor, env2_delay_tensor


def get_station_tensors():
    env1_input_tensors = torch.Tensor(
        torch.rand(4, 4)
    )

    env2_input_tensors = torch.Tensor(
        torch.rand(2, 4)
    )

    return env1_input_tensors, env2_input_tensors


def get_capacity_trains_tensors():
    env1 = torch.Tensor([2])
    env2 = torch.Tensor([2, 3])

    return env1, env2


def get_capacity_station_tensors():
    env1 = torch.Tensor([1, 2, 3])
    env2 = torch.Tensor([3, 2])
    return env1, env2


def get_capacity_route_tensors():
    env1_adj_tensor = torch.Tensor([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ])

    env2_adj_tensor = torch.Tensor([
        [0, 1],
        [1, 0]
    ])

    return env1_adj_tensor, env2_adj_tensor


def get_action_vectors(adj1, adj2):
    pos_train1 = 0
    reachable_stations = adj1[pos_train1]
    destination1 = reachable_stations[1]
    action1 = torch.Tensor([
        [pos_train1, destination1]
    ])

    pos_train21 = 0
    pos_train22 = 1
    reachable_stations1 = adj2[pos_train21]
    reachable_stations2 = adj2[pos_train22]
    destination21 = reachable_stations1[1]
    destination22 = reachable_stations2[1]
    action2 = torch.Tensor([
        [pos_train21, destination21],
        [pos_train22, destination22]
    ])

    return action1, action2


def update_train_pos_routes(length_routes, train_progress):
    mask = train_progress.isnan().logical_not()
    return torch.where(greater_not_close(length_routes * mask, train_progress), 1., float("nan"))


def update_train_pos_stations(length_routes, train_progress):
    mask = train_progress.isnan().logical_not()
    return torch.where(close_or_less(length_routes * mask, train_progress), 1., float("nan"))


def update_capa_station(capa_station, train_pos_stations):
    return capa_station - train_pos_stations.sum(dim=0)  # dims nach batchialisierung um 1 erhöhen!!!!!!


def update_capa_routes(capa_route, train_pos_routes):
    return capa_route - train_pos_routes.sum(dim=0)


def update_train_progress(train_pos_routes, vel, train_progress):
    mask = train_pos_routes.isnan().logical_not()
    train_progress[mask] += vel
    return train_progress


def update_passenger_delay(delay_passenger):
    # increment delay
    return delay_passenger + 1  # torch.add(delay_passenger, 1)


def apply_action(train_progress, length_routes):
    length_routes_w_trains = length_routes * (train_pos_routes.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress, length_routes_w_trains).any(dim=1).any(dim=1)
    train_station = (torch.isnan(train_pos_stations).logical_not() * 1).argmax(dim=2).max(dim=1).values
    reached_train_station = train_station.clone()
    reached_train_station = train_station[train_reached_dest]
    if len(reached_train_station) == 0: return train_station # no possible actions
    possible_actions = torch.cartesian_prod(*adj[reached_train_station])
    possible_actions = possible_actions
    
    # choose action
    action = possible_actions[0]
    #rerouting of trains
    num_reroutable_trains = torch.sum(train_reached_dest)
    new_train_stations = torch.zeros(num_reroutable_trains, train_station.shape[1], train_station.shape[2]) 
    train_station[train_reached_dest]
    return train_station


def onboard_passengers(train_dest, min_dot_req, idx_train, train_station, delay_passenger):
    passenger = torch.where(torch.logical_not(delay_passenger.isnan()), 1, 0)
    passenger_dest = passenger.sum(dim=1).argmax(dim=1)
    passenger_current = torch.transpose(passenger, 1, 2).sum(dim=1).argmax(dim=1)

    swap_tensor1 = torch.LongTensor([passenger_current, idx_train])
    swap_tensor2 = torch.LongTensor([idx_train, passenger_current])
    mask_stations_match = passenger_current == train_station
    mask_passenger_in_train = passenger_current == idx_train
    mask_both = torch.logical_and(mask_stations_match, mask_passenger_in_train)
    swapped_delay_passenger = delay_passenger.clone()
    swapped_delay_passenger[:, swap_tensor1] = swapped_delay_passenger[:, swap_tensor2] # swapping
    delay_passenger = torch.where(mask_both, swapped_delay_passenger, delay_passenger)


    swapped_delay_passenger = delay_passenger.clone()
    swapped_delay_passenger[:, swap_tensor1] = swapped_delay_passenger[:, swap_tensor2] # swapping
    delay_passenger = torch.where(mask_stations_match, swapped_delay_passenger, delay_passenger)

    return delay_passenger


adj, length_routes, _, _ = get_station_adj_routes()
train_progress, vel, _, _ = get_train_tensor()
delay_passenger, _ = get_passenger_tensor()
stations, _ = get_station_tensors()
capa_train, _ = get_capacity_trains_tensors()
capa_station, _ = get_capacity_station_tensors()
capa_route, _ = get_capacity_route_tensors()
action, _ = get_action_vectors(adj, _)

train_pos_routes = update_train_pos_routes(length_routes, train_progress)
train_pos_stations = update_train_pos_stations(length_routes, train_progress)

n_steps = 23
for _ in range(n_steps):
    print(_)
    capa_pos_routes_current = update_capa_routes(capa_route, train_pos_routes)
    capa_station_current = update_capa_station(capa_station, train_pos_stations)
    train_pos_routes = update_train_pos_routes(length_routes, train_progress)
    train_pos_stations = update_train_pos_stations(length_routes, train_progress)
    train_progress = update_train_progress(train_pos_routes, vel, train_progress)
    delay_passenger = update_passenger_delay(delay_passenger)

    train_station = apply_action(train_progress, length_routes)
    
    delay_passenger = onboard_passengers(0, 0, 3, train_station, delay_passenger)
    
    print(delay_passenger)
