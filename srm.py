import torch
import torch.nn.functional as F
import numpy as np


def srm(imgs, device):
    filter1 =  [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
    filter2 = [[0, 0, 0, 0, 0],
               [0, -1, 2, -1, 0],
               [0, 2, -4, 2, 0],
               [0, -1, 2, -1, 0],
               [0, 0, 0, 0, 0]]
    filter3 = [[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0,-2, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]
    filter1 = np.asarray(filter1, dtype=np.float32) / 4.0
    filter2 = np.asarray(filter2, dtype=np.float32) / 12.0
    filter3 = np.asarray(filter3, dtype=np.float32) / 2.0
    filters = np.asarray([[filter1, filter1, filter1], 
               [filter2, filter2, filter2], 
               [filter3, filter3, filter3]], dtype=np.float32)
    filters = torch.tensor(filters, device=device)
    op1 = F.conv2d(imgs, filters, stride=1, padding=2)
    op1 = torch.round(op1)
    op1[op1 > 2] = 2
    op1[op1 < -2] = -2
    return op1
