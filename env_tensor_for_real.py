from torch_scatter import scatter

import utils
import torch
from torch_geometric.utils import to_dense_batch

def init_step(obs):
    adj, capa_station, capa_route, capa_train, train_pos_stations, train_progress, delay_passenger, length_routes, train_pos_routes, vel, vectors = obs
    train_progress = torch.zeros_like(train_pos_stations)
    train_pos_routes, train_pos_stations = update_train_pos(length_routes, train_progress)
    n_stations = len(adj[0])
    print(n_stations)
    pat2s, pas2s = get_possible_actions(adj,
                                        train_progress,
                                        train_pos_stations,
                                        train_pos_routes,
                                        length_routes,
                                        n_stations)
    dones = torch.Tensor([False])
    validities = torch.Tensor([True])
    rewards = torch.Tensor([0])
    return dones, obs, pat2s, pas2s, validities, rewards


def step(actions, obs, logger=None):
    adj, capa_station, capa_route, capa_train, train_pos_stations, train_progress, delay_passenger, length_routes, train_pos_routes, vel, vectors = obs
    n_batches = capa_station.shape[0]
    n_stations = len(adj[0])

    train_progress = update_train_progress(vel, train_progress)
    train_pos_routes, train_pos_stations = update_train_pos(length_routes, train_progress)
    delay_passenger = update_passenger_delay(delay_passenger)

    delay_passenger = onboard_passengers(
        vectors,
        train_progress,
        length_routes,
        train_pos_routes,
        train_pos_stations,
        delay_passenger,
        capa_train,
        n_stations,
        n_batches)

    new_train_pos_routes, new_train_pos_stations, new_train_progress = apply_action(
        actions,
        train_progress,
        length_routes,
        train_pos_routes,
        train_pos_stations,
        n_batches
    )

    train_pos_routes, train_pos_stations, train_progress = new_train_pos_routes, new_train_pos_stations, new_train_progress

    observation = adj, capa_station, capa_route, capa_train, train_pos_stations, train_progress, delay_passenger, length_routes, train_pos_routes, vel, vectors
    possible_actions_train_to_stations, possible_actions_stations_to_stations = get_possible_actions(adj,
                                                                                                     train_progress,
                                                                                                     train_pos_stations,
                                                                                                     train_pos_routes,
                                                                                                     length_routes,
                                                                                                     n_stations)

    capa_station = update_capa_station(capa_station, train_pos_stations)
    capa_route = update_capa_routes(capa_route, train_pos_routes)

    validity = inspect_validity(capa_station, capa_route)
    reward = get_reward(delay_passenger)
    dones = get_dones(delay_passenger)

    return dones, observation, possible_actions_train_to_stations, possible_actions_stations_to_stations, validity, reward


def get_dones(delay_passenger):
    return torch.BoolTensor([False])


def get_reward(delay_passenger):
    return torch.Tensor([0])  # delay_passenger.sum(dim=-1)


def get_possible_actions(adj, train_progress, train_pos_stations, train_pos_routes, length_routes, n_stations):
    train_station = (torch.isnan(train_pos_stations).logical_not()).max(dim=-1).indices.max(dim=-1).values
    length_routesw_trains = length_routes * (train_pos_routes.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress, length_routesw_trains).any(dim=2).any(dim=2)
    reached_train_station = train_station[train_reached_dest]
    train_range = torch.arange(train_progress.shape[1])[None,...][train_reached_dest] + n_stations
    if len(reached_train_station) == 0: return None, None
    a, b = (adj[reached_train_station] == 1).nonzero(as_tuple=True)
    single_possible_actions = to_dense_batch(b, a, fill_value=-1)[0]
    single_possible_actions = single_possible_actions.type(torch.int32)
    # single_t2s_actions = torch.vstack((torch.arange(n_trains), single_possible_actions))
    possible_actions_train_to_stations = torch.cartesian_prod(*single_possible_actions.T)
    possible_actions_stations_to_stations = torch.hstack((train_range, possible_actions_train_to_stations[:, :, 1]))

    return possible_actions_train_to_stations, possible_actions_stations_to_stations


def apply_action(
        actions,
        train_progress,
        length_routes,
        train_pos_routes,
        train_pos_stations,
        n_batches,
        logger=None
):
    if train_pos_stations.isnan().all() or actions is None:  # no possible actions; continue
        return train_pos_routes, train_pos_stations, train_progress
    train_station = (torch.isnan(train_pos_stations).logical_not()).max(dim=-1).indices.max(dim=-1).values
    length_routesw_trains = length_routes * (train_pos_routes.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress, length_routesw_trains).any(dim=2).any(dim=2)
    length_routesw_trains = length_routes * (train_pos_routes.isnan().logical_not())
    row = train_station[train_reached_dest].repeat_interleave(actions.size(dim=-1) // train_reached_dest.size(dim=-1))
    column = actions.argmax(dim=-1, keepdim=True)

    # rerouting of trains
    n_trains = train_pos_routes.shape[-1]
    n_actions = len(actions)
    new_train_stations = torch.zeros(n_batches, n_actions, n_trains, length_routesw_trains.shape[-1],
                                     length_routesw_trains.shape[-2])
    new_train_stations[torch.arange(n_batches), torch.arange(actions.shape[0]), torch.arange(n_trains), row, column] = 1

    new_train_pos_routes = torch.where(new_train_stations == 1, 0., float("nan"))
    new_train_pos_stations = train_pos_stations.clone()
    new_train_pos_stations[...] = float("nan")
    new_train_progress = torch.where(new_train_stations == 1, 0., float("nan"))

    train_pos_routes[train_reached_dest] = new_train_pos_routes
    train_pos_stations[train_reached_dest] = new_train_pos_stations[train_reached_dest]
    train_progress[train_reached_dest] = new_train_progress

    return train_pos_routes, new_train_pos_stations, train_progress


def onboard_passengers(
        vectors,
        train_progress,
        length_routes,
        train_pos_routes,
        train_pos_stations,
        delay_passenger,
        train_capa,
        n_stations,
        logger=None
):
    """
        This function applies the de- and onboarding of the passengers from and to trains.
        This is done by swapping the rows of the delaymatrix from the index of the current station/train to the
        destination train/station.
    """
    if train_pos_stations.isnan().all():  # no train in station
        return delay_passenger

    length_routesw_trains = length_routes * (train_pos_routes.isnan().logical_not())
    train_reached_dest = greater_not_close(train_progress, length_routesw_trains).any(dim=2).any(dim=2)
    train_station = (torch.isnan(train_pos_stations).logical_not()).max(dim=-1).indices.max(dim=-1).values
    train_dest = (torch.isnan(torch.transpose(train_pos_stations, 1, 2)).logical_not()).max(dim=-1).indices.max(
        dim=-1).values

    n_trains = train_pos_routes.shape[1]
    n_passenger = delay_passenger.shape[1]

    # deboard all pasengers from trains, in order to only have the most "important" passengers on board
    delay_passenger = boarding(vectors, train_reached_dest, delay_passenger, train_station, train_dest, train_capa, n_trains, n_passenger, n_stations, onboarding=False)
    print("deboarded")
    # onboard
    delay_passenger = boarding(vectors, train_reached_dest, delay_passenger, train_station, train_dest, train_capa, n_trains, n_passenger, n_stations, onboarding=True)
    print("onboarded")
    return delay_passenger


def boarding(vectors, train_reached_dest, delay_passenger, train_station, train_dest, train_capa, n_trains, n_passenger, n_stations, onboarding=False):
    passenger = torch.where(delay_passenger.isnan(), 0, 1)
    passenger_current = torch.transpose(passenger, 2, 3).sum(dim=2).argmax(dim=2, keepdim=True).long().flatten(
        start_dim=1)
    passenger_dest = passenger.sum(dim=2).argmax(dim=2, keepdim=True).long().flatten(start_dim=1)
    passenger_matrices = passenger_current[..., None].expand(-1, -1, n_trains)

    if onboarding:
        train_station = torch.where(train_reached_dest, train_station, -1)
        train_matrices = train_station[..., None].expand(-1, -1, n_passenger)

    else: # passenger is in train
        train_range = torch.arange(n_trains) + n_stations
        train_range = torch.where(train_reached_dest, train_range, -1)
        train_matrices = train_range[..., None].expand(-1, -1, n_passenger)

    req = passenger_matrices.transpose(1, 2) == train_matrices
    if not req.any(): # no passenger fulfills requirement
        return delay_passenger # change nothing
    batch_req, train_req, pass_req = req.nonzero(as_tuple=True)

    if onboarding:
        vectors_dest_train = vectors[batch_req, train_dest[batch_req, train_req]]
        vectors_dest_pass = vectors[batch_req, passenger_dest[batch_req, pass_req]]
        vectors_current = vectors[batch_req, passenger_current[batch_req, pass_req]]
        advantage = torch.einsum('ij,ij->i', vectors_dest_train - vectors_current, vectors_dest_pass)
        mask = (advantage > 0) # mask lacking capas

    else:
        mask = torch.BoolTensor([True]) # everyone has to deboard

    dest_pass = passenger_dest[batch_req, pass_req]
    delay_pass = delay_passenger[batch_req, pass_req][mask]
    delay_pass = delay_pass[~torch.isnan(delay_pass)]
    new_delay_passenger = delay_passenger
    new_delay_passenger[batch_req, pass_req][mask][...] = float("NaN")
    new_delay_passenger[batch_req, pass_req, train_req, dest_pass][mask] = delay_pass

    return new_delay_passenger


def update_train_pos(length_routes, train_progress):
    is_on_route = train_progress.isnan().logical_not()
    train_pos_routes = torch.where(greater_not_close(length_routes * is_on_route, train_progress), 1., float("nan"))
    train_pos_stations = torch.where(close_or_less(length_routes * is_on_route, train_progress), 1., float("nan"))
    return train_pos_routes, train_pos_stations


def update_capa_station(capa_station_, train_pos_stations):
    return capa_station_ - torch.where(train_pos_stations.isnan(), torch.tensor(0).float(), train_pos_stations).sum(
        dim=1)


def update_capa_routes(capa_route, train_pos_routes):
    return capa_route - torch.where(train_pos_routes.isnan(), torch.tensor(0).float(), train_pos_routes).sum(dim=1)


def update_train_progress(vel, train_progress):
    return train_progress + vel[..., None, None]


def update_passenger_delay(delay_passenger):
    return delay_passenger + 1


def inspect_validity(capa_station, capa_routes):
    return (torch.cat((capa_station, capa_routes)) >= 0).any()


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
