import torch
y = torch.empty((3,)) #torch.tensor([])
print(y)
y = torch.vstack((y, torch.tensor([1,2 ,3])))
print(y)
y = torch.vstack((y, torch.tensor([1,2 ,3])))#.unsqueeze(-1)))


print(y)