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


def get_station_adj_routes():
    env1_adj_tensor = torch.Tensor([
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1]
    ])
    length_routes1 = torch.Tensor([
        [0, 3, 0],
        [0, 0, 0],
        [0, 4, 0]
    ])
    env2_adj_tensor = torch.Tensor([
        [1, 1],
        [1, 1]
    ])
    length_routes2 = torch.Tensor([
        
    ])
    return env1_adj_tensor, length_routes1,env2_adj_tensor, length_routes2

def get_train_tensor():
    env1_train_tensor = torch.Tensor([
        [
            [float("NaN"), .3, float("NaN")],
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
    env1_delay_tensor = torch.Tensor([
        [float('nan'), float('nan'), float('nan')],
        [float('nan'), float('nan'), float('nan')],
        [float('nan'), float('nan'), -12]
    ])

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
    torch.rand(3, 4)
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
    mask = train_progress > 0
    return torch.where(length_routes*mask>train_progress, 1., float("nan"))

def update_train_pos_stations(length_routes, train_progress):
    mask = train_progress > 0
    return torch.where(length_routes*mask<train_progress, 1.,  float("nan"))

def update_capa_station(capa_station, train_pos_stations):
   return capa_station - train_pos_stations.sum(dim=0) # dims nach batchialisierung um 1 erhöhen!!!!!!

def update_capa_routes(capa_route, train_pos_routes):
    return capa_route - train_pos_routes.sum(dim=0)

def update_train_progress(train_pos_routes, vel, train_progress):
    train_progress[train_pos_routes>0] += vel
    return train_progress

def update_passenger_delay(delay_passenger):
    # increment delay
    return torch.add(delay_passenger, 1)

def apply_action(train_progress, length_routes):
    if ((train_progress*1_000).round()/1_000 >= length_routes*(train_pos_routes.isnan().logical_not())).any(): 
        print("ankomme")
    print(train_progress*(train_pos_routes>0))
    print( length_routes*(train_pos_routes>0))
    print()

adj, length_routes, _, _ = get_station_adj_routes()
train_progress, vel, _, _ = get_train_tensor()
train_pos_routes = torch.where(length_routes>train_progress, 1., float("NaN"))
train_pos_stations = torch.where(train_progress>length_routes, 1., float("nan"))
delay_passenger, _ = get_passenger_tensor()
stations, _ = get_station_tensors()
capa_train, _ = get_capacity_trains_tensors()
capa_station, _ = get_capacity_station_tensors()
capa_route, _ = get_capacity_route_tensors()
action,  _ = get_action_vectors(adj, _)

train_pos_routes = update_train_pos_routes(length_routes, train_progress)
train_pos_stations = update_train_pos_stations(length_routes, train_progress)

n_steps = 31
for _ in range(n_steps):


    capa_pos_routes_current = update_capa_routes(capa_route, train_pos_routes)
    capa_station_current = update_capa_station(capa_station, train_pos_stations)
    train_pos_routes = update_train_pos_routes(length_routes, train_progress)
    train_pos_stations = update_train_pos_stations(length_routes, train_progress)
    train_progress = update_train_progress(train_pos_routes, vel, train_progress)
    delay_passenger += update_passenger_delay(delay_passenger)

    apply_action(train_progress, length_routes)