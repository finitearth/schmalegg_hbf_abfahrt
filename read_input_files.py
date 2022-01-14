from cProfile import label
import re
import torch
from objects import Station, Routes, PassengerGroup, Train
import networkx  as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def read_txt():
    file_path = str(input())
    with open(file_path, 'r') as f:
        text = f.read()

    text = text.lower()
    text = "".join([t + "\n" for t in text.split("\n") if "#" not in t])
    splits = re.split(r"(\[stations\]|\[lines\]|\[passengers\]|\[trains\])", text)

    for i, split in enumerate(splits):
        if "[stations]" in split:
            single_stations = splits[i + 1].replace("[stations]\n", "").split('\n') 
        elif "[lines]" in split:
            single_lines = splits[i + 1].replace("[lines]\n", "").split('\n') 
        elif "[passengers]" in split:
            single_passengers = splits[i + 1].replace("[passengers]\n", "").split('\n') 
        elif "[trains]" in split:
            single_trains = splits[i + 1].replace("[trains]\n", "").split('\n') 


    #Stations
    station_capa = []
    station_name = []
    
    for i, s in enumerate([s for s in single_stations if s != "" and s != " "]):
        ss = s.split(" ")
        station_name.append(ss[0])
        capacity = int(ss[1])
        station_capa.append(capacity)
    station_capa_tensor = torch.tensor(station_capa).unsqueeze(0)


    #Routes
    g = nx.Graph()
    capa_route_list = []
    for l in single_lines:
        if l == "" or l == " ": continue
        ll = l.split(" ")
        capa_route_list.append((ll[1],ll[2],ll[4]))
        g.add_edge(ll[1], ll[2], weight=-float(ll[3]))

    d = nx.spring_layout(g, dim=4, threshold=0.01)
    vectors = np.array([v for v in d.values()])
    vectors = torch.Tensor(vectors)
    length_routes = nx.to_pandas_adjacency(g)
    length_routes = length_routes.sort_index(axis=0)[station_name]
    length_routes = torch.Tensor(length_routes.values)
    adj = torch.where(length_routes==0, 0, 1)

    capa_routes =  torch.zeros_like(length_routes).unsqueeze(0)
    for s1, s2, capa in capa_route_list:
        capa_routes[0, station_name.index(s1), station_name.index(s2)] = int(capa)
        capa_routes[0, station_name.index(s2), station_name.index(s1)] = int(capa)


    #Trains
    train_list = []
    velocity = []
    capacity = []
    for t in single_trains:
        if t == "" or t == " " or '*' in t: continue
        tt = t.split(" ")
        train_list.append(station_name.index(tt[1]))
        velocity.append(float(tt[2]))
        capacity.append(int(tt[3]))

    vel =  torch.tensor(velocity)
    cap_tensor =  torch.tensor(capacity)

    train_pos_stations = torch.zeros((1,len(train_list),len(station_name),len(station_name)))
    train_pos_stations[...] = float("nan")
    for i, s in enumerate(train_list):
        train_pos_stations[0, i, s, 0] = 0 
    
    train_pos_routes = torch.zeros_like(train_pos_stations)
    train_pos_routes[...] = float("nan")
    # train with * variable position -> maybe OpenAI Hide and Seek? 

    #Trains
    passenger = []
    for p in single_passengers:
        if p == "" or p == " ": continue
        pp = p.split(" ")
        passenger.append((pp[1],pp[2],pp[4]))
    n_stations = len(station_name)
    passenger_delay = torch.zeros((1, len(passenger), (n_stations+len(train_list)), n_stations))
    for i, (s1, s2, delay) in enumerate(passenger):
        j = station_name.index(s1)
        k = station_name.index(s2)
        passenger_delay[0, i, j, k] = -float(delay)

    print(file_path)
    return vectors, adj, station_capa_tensor, capa_routes, cap_tensor, train_pos_stations, length_routes, vel, passenger_delay