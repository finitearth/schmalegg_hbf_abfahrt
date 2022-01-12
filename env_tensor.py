import torch
import utils

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
        [0, 1, 1, 0, 0],  # station1
        [1, 0, 1, 0, 0],  # station2
        [0, 1, 0, 0, 1],  # station3
        [1, 1, 0, 0, 0],  # station4
        [1, 0, 0, 1, 0]  # station5
        

    ])
    length_routes1 = torch.Tensor([
        [1, 3, 2, 3, 2],
        [2, 1, 2, 3, 7],
        [5, 4, 1, 5, 2],
        [5, 4, 1, 3, 1],
        [5, 4, 1, 9, 8]
    ])
    return env1_adj_tensor, length_routes1


def get_train_tensor():
    env1_train_tensor = torch.Tensor([[
        [
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), 0, float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")]
        ],
        [
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), 0, float("NaN"), float("NaN"), float("NaN")]
        ],

        [
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), 0, float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")]
        ],
        [
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), 0, float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")],
            [float("NaN"), float("NaN"), float("NaN"), float("NaN"), float("NaN")]
        ]
    ]])

    env1_trains_velocity = torch.Tensor([0.88, 0.1, 0.5, 0.23 ])

    return env1_train_tensor, env1_trains_velocity


def get_passenger_tensor():
    env1_delay_tensor = torch.Tensor([[
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), -12, float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

        [[float('nan'), float('nan'), -30, float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), -5, float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), -40, float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), -20, float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), -18, float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), -100, float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), -30, float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), -50, float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [-17, float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), -33, float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[-15, float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[-69, float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), -88, float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), -36, float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), -12, float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), -10],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'),-11],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [-88, float('nan'), float('nan'), float('nan'), float('nan')],  # Spalten: Ziel
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],  # Zeile: aktueller Bahnhof
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],
         
        [[float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), -47, float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')],
         [float('nan'), float("nan"), float('nan'), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')],
         [float("nan"), float("nan"), float("nan"), float('nan'), float('nan')]],

         
         
    ]])

    return env1_delay_tensor


def get_station_tensors():
    env1_input_tensors = torch.Tensor(
        torch.rand(9, 4)
    )

    return env1_input_tensors


def get_capacity_trains_tensors():
    env1 = torch.Tensor([[2, 20, 10, 5]])

    return env1


def get_capacity_station_tensors():
    env1 = torch.Tensor([[1, 2, 3, 10, 10]])
    return env1


def get_capacity_route_tensors():
    env1_adj_tensor = torch.Tensor([[
        [0, 1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ]])

    return env1_adj_tensor


def update_train_pos(length_routes_, train_progress_):
    is_on_route = train_progress.isnan().logical_not()
    train_pos_routes_ = torch.where(greater_not_close(length_routes_*is_on_route, train_progress_), 1., float("nan"))
    train_pos_stations_ = torch.where(close_or_less(length_routes_*is_on_route, train_progress_), 1., float("nan"))
    return train_pos_routes_, train_pos_stations_


def update_capa_station(capa_station_, train_pos_stations_):
    return capa_station_ - train_pos_stations_.sum(dim=1)


def update_capa_routes(capa_route_, train_pos_routes_):
    return capa_route_ - train_pos_routes_.sum(dim=1)


def update_train_progress(vel_, train_progress_):
    return train_progress_ + vel_[..., None, None]


def update_passenger_delay(delay_passenger_):
    return delay_passenger_ + 1


def onboard_passengers(train_progress_, length_routes_, train_pos_routes_, delay_passenger_):
    if train_pos_stations.isnan().all():  # no train in station
        return delay_passenger_
    length_routes_w_trains = length_routes_ * (train_pos_routes_.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress_, length_routes_w_trains).any(dim=2).any(dim=2)
    train_station_ = (torch.isnan(train_pos_stations).logical_not()).max(dim=-1).indices.max(dim=-1).values
    passenger = torch.where(delay_passenger_.isnan(), 0, 1)
    train_range = torch.arange(train_pos_routes.shape[1])[None, ...]
    idx_train = train_range[train_reached_dest].long() + n_stations

    passenger_current = torch.transpose(passenger, 2, 3).sum(dim=2).argmax(dim=2).long()
    passenger_dest = passenger.sum(dim=2).argmax(dim=2).long()
    reached_passenger = passenger_current == passenger_dest

    n_passenger_current = passenger_current.shape[-1]
    n_idx_train = idx_train.shape[-1]

    idx_train = torch.repeat_interleave(idx_train, max(1, n_passenger_current // n_idx_train))
    passenger_current = torch.repeat_interleave(passenger_current, max(1, n_idx_train // n_passenger_current))

    swap_tensor1 = torch.vstack((passenger_current, idx_train))[None, ...]
    swap_tensor2 = torch.vstack((idx_train, passenger_current))[None, ...]
    range_passenger = torch.arange(len(passenger_current))
    mask_passenger_in_train = (passenger_current == idx_train)
    swapped_delay_passenger = delay_passenger_.clone()
    swapped_delay_passenger[:, range_passenger, swap_tensor1] = swapped_delay_passenger[:,
                                                                range_passenger, swap_tensor2]  # aussteigen
    delay_passenger_[:, mask_passenger_in_train] = swapped_delay_passenger[:, mask_passenger_in_train]
    #train_station_ = torch.repeat_interleave(train_station_, max(1, n_passenger_current // train_station_.shape[-1]))
    train_reached_dest = greater_not_close(train_progress_, length_routes_w_trains).any(dim=2).any(dim=2)
    train_reached_station_ = train_station_[train_reached_dest]
    mask_stations_match = (passenger_current == train_reached_station_) 
    swapped_delay_passenger = delay_passenger_.clone()
    swapped_delay_passenger[:, range_passenger, swap_tensor1] = swapped_delay_passenger[:,
                                                                range_passenger, swap_tensor2]  # einsteigen
    delay_passenger_[:, mask_stations_match] = swapped_delay_passenger[:, mask_stations_match]

    return delay_passenger_[None, torch.logical_not(reached_passenger)]


def apply_action(train_progress_, length_routes_, train_pos_routes_, train_pos_stations_):
    if train_pos_stations_.isnan().all():  # no possible actions; continue
        return train_pos_routes_, train_pos_stations_, train_progress_
    train_station = (torch.isnan(train_pos_stations).logical_not()).max(dim=-1).indices.max(dim=-1).values
    length_routes_w_trains = length_routes_ * (train_pos_routes_.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress_, length_routes_w_trains).any(dim=2).any(dim=2)
    reached_train_station = train_station[train_reached_dest]
    possible_actions = utils.cart_prod(adj[reached_train_station]*torch.arange(n_stations))
    row = train_station[train_reached_dest].repeat_interleave(
        possible_actions.size(dim=-1) // train_reached_dest.size(dim=-1))
    column = possible_actions.argmax(dim=-1, keepdim=True)

    # rerouting of trains
    n_trains = len(reached_train_station)
    n_actions = len(possible_actions)
    n_batches = 1
    new_train_stations = torch.zeros(n_batches, n_actions, n_trains, length_routes_w_trains.shape[-1],
                                     length_routes_w_trains.shape[-2])
    new_train_stations[:, torch.arange(possible_actions.shape[0]), torch.arange(n_trains), row, column] = 1
    new_train_station = new_train_stations[:, 0]  # choose action
    new_train_pos_routes_ = torch.where(new_train_station == 1, 0., float("nan"))
    new_train_pos_stations_ = train_pos_stations.clone()
    new_train_pos_stations_[...] = float("nan")
    new_train_progress_ = torch.where(new_train_station == 1, 0., float("nan"))

    train_pos_routes_[train_reached_dest] = new_train_pos_routes_
    train_pos_stations_[train_reached_dest] = new_train_pos_stations_[train_reached_dest]
    train_progress_[train_reached_dest] = new_train_progress_

    return train_pos_routes_, train_pos_stations_, train_progress_


adj, length_routes = get_station_adj_routes()  # beides konstant
stations = get_station_tensors()  # stations konstant
train_progress, vel = get_train_tensor()  # vel konstant
delay_passenger = get_passenger_tensor()  # variabel
capa_train = get_capacity_trains_tensors()
capa_station = get_capacity_station_tensors()
capa_route = get_capacity_route_tensors()

train_pos_routes, train_pos_stations = update_train_pos(length_routes, train_progress)
n_stations = 5
n_steps = 1_000

for _ in range(n_steps):
    # print(_)
    #capa_pos_routes_current = update_capa_routes(capa_route, train_pos_routes)
    capa_station_current = update_capa_station(capa_station, train_pos_stations)
    train_pos_routes, train_pos_stations = update_train_pos(length_routes, train_progress)
    train_progress = update_train_progress(vel, train_progress)
    delay_passenger = update_passenger_delay(delay_passenger)

    delay_passenger = onboard_passengers(train_progress, length_routes, train_pos_routes, delay_passenger)

    train_pos_routes, train_pos_stations, train_progress = apply_action(train_progress, length_routes,
                                                                        train_pos_routes,
                                                                        train_pos_stations)

                     

#  TODO mask out full capas
