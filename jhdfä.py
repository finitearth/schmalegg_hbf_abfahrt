import torch.nn.functional as F
import torch
starts = torch.zeros(1, 4, 4, 2)
n_stations = 4
actions = torch.tensor([[[1, 2, 3], [1, 1, 1]]])
start_vecs = torch.rand(starts.shape[0], starts.shape[1], 2)

starts = start_vecs[:, actions[:, 0, :]]


print(starts)