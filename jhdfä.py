import torch


actions = torch.Tensor([1, 2, 3])
best_action = torch.Tensor([1])
one_hot = (actions == best_action)*1#.nonzero(as_tuple=True)
# one_hot = torch.zeros_like(actions)
# one_hot[best_id] = 1
print(one_hot)
# print(best_id)