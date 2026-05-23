# Attempt 1 — 22 May, 2026
# Started: 10:20pm
# finished: 10:27pm
# finished after correction hints: 10:40
# First pass (runs without error): yes
# Correct (passes tests): partial
# Stuck on: none
# Notes: missed adding axis for max initally

import numpy as np
import torch

def compute_softmax(x, axis=-1):
    #print(x.shape, axis)
    max_per_axis = np.max(x, axis, keepdims=True)
    #print(max_per_axis.shape)
    exp = np.exp(x - max_per_axis) 
    #print(exp.shape)
    axis_sum = np.sum(exp, axis=axis, keepdims=True)
    softmax = exp/axis_sum
    #print(softmax.shape)
    return softmax

x = np.random.randn(2,4,5)
axis = 1
out_1 = compute_softmax(x, axis)

x2 = np.array([[1000, 1001, 1002], [2, 5, 8]], dtype=float)
out_2 = compute_softmax(x2)
print(torch.allclose(torch.from_numpy(out_1), torch.softmax(torch.from_numpy(x), dim=axis)))
print(torch.allclose(torch.from_numpy(out_2), torch.softmax(torch.from_numpy(x2), dim=axis)))