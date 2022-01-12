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
        [0, 1, 1],  # station1
        [1, 0, 1],  # station2
        [1, 1, 0],  # station3
        [0, 0, 0]  # train1

    ])
    length_routes1 = torch.Tensor([
        [1, 3, 2],
        [2, 1, 2],
        [5, 4, 1]
    ])
    return env1_adj_tensor, length_routes1


def get_train_tensor():
    env1_train_tensor = torch.Tensor([
        [
            [float("NaN"), float("NaN"), 0],
            [float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN")]
        ],
        [
            [float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), 0, float("NaN")]
        ]
    ])

    env1_trains_velocity = torch.Tensor([0.88, 0.1])

    return env1_train_tensor, env1_trains_velocity


def get_passenger_tensor():
    env1_delay_tensor = torch.Tensor([
        [[float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), -12, float('nan')],
         [float("nan"), float("nan"), float("nan")],
         [float("nan"), float("nan"), float("nan")]],
        [[float('nan'), float('nan'), -30],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float("nan"), float('nan')],
         [float("nan"), float("nan"), float("nan")],
         [float("nan"), float("nan"), float("nan")]]
    ])

    return env1_delay_tensor


def get_station_tensors():
    env1_input_tensors = torch.Tensor(
        torch.rand(5, 4)
    )

    return env1_input_tensors


def get_capacity_trains_tensors():
    env1 = torch.Tensor([2, 20])

    return env1


def get_capacity_station_tensors():
    env1 = torch.Tensor([1, 2, 3])
    return env1


def get_capacity_route_tensors():
    env1_adj_tensor = torch.Tensor([
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ])

    return env1_adj_tensor


def update_train_pos(length_routes_, train_progress_):
    mask = train_progress.isnan().logical_not()
    train_pos_routes_ = torch.where(greater_not_close(length_routes_ * mask, train_progress_), 1., float("nan"))
    train_pos_stations_ = torch.where(close_or_less(length_routes_ * mask, train_progress_), 1., float("nan"))
    return train_pos_routes_, train_pos_stations_


def update_capa_station(capa_station_, train_pos_stations_):
    return capa_station_ - train_pos_stations_.sum(dim=0)  # dims nach batchialisierung um 1 erhöhen!!!!!!


def update_capa_routes(capa_route, train_pos_routes):
    return capa_route - train_pos_routes.sum(dim=0)


def update_train_progress(vel_, train_progress_):
    train_progress_ = (train_progress_.T + vel_).T
    return train_progress_


def update_passenger_delay(delay_passenger):
    # increment delay
    return delay_passenger + 1  # torch.add(delay_passenger, 1)


def onboard_passengers(train_progress_, length_routes_, train_pos_routes_, delay_passenger_):
    if train_pos_stations.isnan().all():  # no train in station
        return delay_passenger_
    length_routes_w_trains = length_routes_ * (train_pos_routes_.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress_, length_routes_w_trains).any(dim=1).any(dim=1)
    train_station_ = (torch.isnan(train_pos_stations).logical_not() * 1).argmax(dim=2).max(dim=1).values

    # reached_train_station = train_station_.clone()
    # reached_train_station = reached_train_station[train_reached_dest]
    passenger = torch.where(delay_passenger_.isnan(), 0, 1)
    # passenger_dest = passenger.sum(dim=1).argmax(dim=1)
    print((torch.arange(train_pos_routes.shape[0])[train_reached_dest]).long())
    idx_train = (torch.arange(train_pos_routes.shape[0])[train_reached_dest]).long() + n_stations

    passenger_current = torch.transpose(passenger, 1, 2).sum(dim=1).argmax(dim=1).long()

    n_passenger_current = passenger_current.shape[-1]
    n_idx_train = idx_train.shape[-1]

    idx_train = torch.repeat_interleave(idx_train, max(1, n_passenger_current // n_idx_train))
    passenger_current = torch.repeat_interleave(passenger_current, max(1, n_idx_train // n_passenger_current))

    swap_tensor1 = torch.vstack((passenger_current, idx_train))  # torch.LongTensor([passenger_current, idx_train])
    swap_tensor2 = torch.vstack((idx_train, passenger_current))
    range_passenger = torch.arange(len(passenger_current))
    mask_passenger_in_train = torch.BoolTensor(passenger_current == idx_train)
    swapped_delay_passenger = delay_passenger_.clone()
    swapped_delay_passenger[range_passenger, swap_tensor1] = swapped_delay_passenger[
        range_passenger, swap_tensor2]  # aussteigen
    delay_passenger_[mask_passenger_in_train] = swapped_delay_passenger[
        mask_passenger_in_train]  # torch.where(mask_passenger_in_train, swapped_delay_passenger, delay_passenger_)

    mask_stations_match = torch.BoolTensor(passenger_current == train_station_)
    swapped_delay_passenger = delay_passenger_.clone()
    swapped_delay_passenger[range_passenger, swap_tensor1] = swapped_delay_passenger[
        range_passenger, swap_tensor2]  # einsteigen
    delay_passenger_[mask_stations_match] = swapped_delay_passenger[mask_stations_match]

    return delay_passenger_


def apply_action(train_progress_, length_routes_, train_pos_routes_, train_pos_stations_):
    if train_pos_stations_.isnan().all():  # no possible actions; continue
        return train_pos_routes_, train_pos_stations_, train_progress_
    train_station = (torch.isnan(train_pos_stations).logical_not() * 1).argmax(dim=2).max(dim=1).values
    length_routes_w_trains = length_routes_ * (train_pos_routes_.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress_, length_routes_w_trains).any(dim=1).any(dim=1)
    reached_train_station = train_station[train_reached_dest]
    possible_actions = torch.cartesian_prod(*adj[reached_train_station])
    boolean_tensor = torch.where(possible_actions == 1, True, False)
    possible_actions = torch.arange(possible_actions.shape[-1])[boolean_tensor]
    row = train_station[train_reached_dest].repeat_interleave(
        possible_actions.size(dim=-1) // train_reached_dest.size(dim=-1))
    column = possible_actions

    # rerouting of trains
    n_trains = len(reached_train_station)
    n_actions = len(possible_actions)
    new_train_stations = torch.zeros(n_actions, n_trains, length_routes_w_trains.shape[1],
                                     length_routes_w_trains.shape[2])
    new_train_stations[torch.arange(possible_actions.shape[0]), torch.arange(n_trains), row, column] = 1
    new_train_station = new_train_stations[0]  # choose action
    new_train_pos_routes_ = torch.where(new_train_station == 1, 0., float("nan"))
    new_train_pos_stations_ = train_pos_stations.clone()
    new_train_pos_stations_[...] = float("nan")
    new_train_progress_ = torch.where(new_train_station == 1, 0., float("nan"))

    # train_station_[train_reached_dest] = new_train_station
    train_pos_routes_[train_reached_dest] = new_train_pos_routes_  # [train_reached_dest]
    train_pos_stations_[train_reached_dest] = new_train_pos_stations_[train_reached_dest]
    train_progress_[train_reached_dest] = new_train_progress_  # [train_reached_dest]

    return train_pos_routes_, train_pos_stations_, train_progress_


adj, length_routes = get_station_adj_routes()  # beides konstant
train_progress, vel = get_train_tensor()  # vel konstant
delay_passenger = get_passenger_tensor()  # nix konstant
stations = get_station_tensors()  # stations konstant
capa_train = get_capacity_trains_tensors()  # variabel
capa_station = get_capacity_station_tensors()  # variabel
capa_route = get_capacity_route_tensors()  # variabel

train_pos_routes, train_pos_stations = update_train_pos(length_routes, train_progress)
n_stations = 3
n_steps = 100
for _ in range(n_steps):
    # print(_)
    capa_pos_routes_current = update_capa_routes(capa_route, train_pos_routes)
    capa_station_current = update_capa_station(capa_station, train_pos_stations)
    train_pos_routes, train_pos_stations = update_train_pos(length_routes, train_progress)
    train_progress = update_train_progress(vel, train_progress)
    delay_passenger = update_passenger_delay(delay_passenger)

    delay_passenger = onboard_passengers(train_progress, length_routes, train_pos_routes, delay_passenger)

    train_pos_routes, train_pos_stations, train_progress = apply_action(train_progress, length_routes,
                                                                        train_pos_routes,
                                                                        train_pos_stations)

#  TODO mask out full capas
