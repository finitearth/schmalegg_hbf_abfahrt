from cProfile import label
import re
import torch
from objects import Station, Routes, PassengerGroup, Train
import networkx  as nx
import matplotlib.pyplot as plt

def read_txt(file_path):

    with open(file_path, 'r') as f:
        text = f.read()

    text = text.lower()
    text = "".join([t + "\n" for t in text.split("\n") if "#" not in t])
    splits = re.split(r"(\[stations\]|\[lines\]|\[passengers\]|\[trains\])", text)

    for i, split in enumerate(splits):
        if "[stations]" in split:
            stations_text = splits[i + 1].replace("[stations]\n", "")
        elif "[lines]" in split:
            lines_text = splits[i + 1].replace("[lines]\n", "")
        elif "[passengers]" in split:
            passengers_text = splits[i + 1].replace("[passengers]\n", "")
        elif "[trains]" in split:
            trains_text = splits[i + 1].replace("[trains]\n", "")

    single_stations = stations_text.split('\n')
    station_list = []
    stations_dict = {}
    station_capa = []
    
    for i, s in enumerate([s for s in single_stations if s != "" and s != " "]):
        ss = s.split(" ")
        station_list.append(ss)
        stations_dict[ss[0]] = ss
        capacity = int(ss[1])
        station_capa.append(capacity)
    
    station_tensor_size = len(station_list)
    stations_tensor = torch.zeros(station_tensor_size, station_tensor_size)
    print(station_capa)

    routes = Routes()
    single_lines = lines_text.split('\n')#
    g = nx.Graph()
    for l in single_lines:
        if l == "" or l == " ": continue
        ll = l.split(" ")
        g.add_edge(ll[1], ll[2], label=ll[0], weight=-float(ll[3]))
    d = nx.spring_layout(g, dim=4)
    nx.draw(g)
    plt.show()          


    # single_trains = trains_text.split('\n')
    # train_list = []
    # for t in single_trains:
    #     if t == "" or t == " ": continue
    #     tt = t.split(" ")
    #     ttt = stations_dict[tt[1]] if tt[1] != "*" else list(stations_dict.values())[0]
    #     train_list.append(Train(ttt, int(tt[3]), name=int(tt[0][1:])))

    # passengers_text = passengers_text.split('\n')
    # passenger = []
    # for p in passengers_text:
    #     if p == "" or p == " ": continue
    #     pp = p.split(" ")
    #     ppp = PassengerGroup(start_station=stations_dict[pp[1]], destination=stations_dict[pp[2]], n_people=pp[3],
    #                          target_time=pp[4])
    #     passenger.append(ppp)
    #     stations_dict[pp[1]].passengers.append(ppp)
    

read_txt("input\input5.txt")