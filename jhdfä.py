import torch

import utils

import time

t0 = time.perf_counter()

for _ in range(1):
    a = torch.Tensor([[1, 2, 3], [1, 2, 3]])
    falsches_hoffentlich_nicht = utils.maybe_performant_cart_product_vlt_maybe(a)
    # richtiges = (torch.cartesian_prod(*a))

print(time.perf_counter() - t0)

# 

# print(richtiges == falsches_hoffentlich_nicht)

# print(falsches_hoffentlich_nicht)

# print(richtiges)