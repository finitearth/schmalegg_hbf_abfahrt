# import torch
# import torch.nn as F
#
# dest = torch.tensor([[[ 2.0430e-02,  3.8745e-01,  3.5094e-01,  3.4751e-02, -9.6034e-02,
#           -6.3019e-01,  7.2855e-01, -8.3846e-03],
#          [ 3.2951e-04,  3.8314e-01,  3.0703e-01,  4.6401e-02, -1.5823e-01,
#           -6.0388e-01,  6.8322e-01,  1.2412e-02],
#          [-9.3841e-02,  4.1911e-01,  9.8503e-01, -4.3419e-02, -3.7862e-01,
#           -1.3768e+00,  1.1794e+00,  2.8229e-02],
#          [ 7.2022e-02,  3.6887e-01, -1.5405e-01,  1.6997e-01,  6.3094e-02,
#           -5.9964e-02,  3.2629e-01,  2.3761e-02]]])
#
# start = torch.tensor([[[ 0.0221,  0.3869,  0.3405,  0.0360, -0.0929, -0.6181,  0.7212,
#           -0.0090],
#          [-0.0791,  0.4199,  1.0153, -0.0524, -0.3328, -1.3908,  1.2098,
#            0.0110],
#          [-0.1070,  0.5437,  1.6615, -0.0627, -0.3359, -2.0648,  1.7497,
#           -0.0730],
#          [-0.0769,  0.4228,  1.0275, -0.0518, -0.3250, -1.4015,  1.2207,
#            0.0077]]])
#
# actions = torch.tensor([[[[0, 3]]]])
# # actions = actions.swapaxes(0, 1)
# print(actions[:, :, :, 0].flatten())
# # print(start)
# # print(actions)
#
# starts = start[:, actions[:, :, :, 0].flatten()]#.unsqueeze(0)#  # batches, action, stations, bool_starting
# dests = dest[:, actions[:, :, :, 1].flatten()]#.unsqueeze(0)#]
#
# print(starts)
# print(dests)
#
# probs = torch.einsum('bij,bij->bi', starts, dests)
# # probs = starts @ dests
# m = F.Softmax(dim=1)
# probs = m(probs)
# print(probs)

total_memory = 8000000000 # bytes
bytes_per_element = 4
elements_per_station = 8
actions = (171//4)**5

percent_used = bytes_per_element * elements_per_station * actions * 2 / total_memory * 100

print(percent_used)