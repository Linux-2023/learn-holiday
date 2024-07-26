import torch
import torch.nn as nn
from params import *
"""
Pnn

question embed:     q
average skill:      s
attribute feature:  a

input:  Z = (z1, z2, z3) = (q, s, a)
        P = [pij], pij=<zi, zj>

transform 2 info matrix -> signal vector lz, lp

l_z^k = sum(W_z^k * Z)
l_p^k = sum(W_P ^k * P)

W_p^k = theta @ theta.T

e = ReLU(l_z + l_p + b)

"""
class PNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, keep_prob):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim * 3 + 3, hidden_dim).to(DEVICE)
        self.linear2 = nn.Linear(hidden_dim, 1).to(DEVICE)
        self.act = nn.ReLU().to(DEVICE)
        self.dropout = nn.Dropout(p=keep_prob).to(DEVICE)
        self.embed_dim = embed_dim

    def forward(self, inputs):
        num_inputs = len(inputs)
        num_pairs = int(num_inputs * (num_inputs - 1) / 2)
        xw = torch.cat(inputs, 1)
        xw3d = xw.reshape(-1, num_inputs, self.embed_dim)

        row, col = [], []
        for i in range(num_inputs - 1):
            for j in range(i + 1, num_inputs):
                row.append(i)
                col.append(j)

        p = xw3d.permute(1, 0, 2)[row].permute(1, 0, 2)
        q = xw3d.permute(1, 0, 2)[col].permute(1, 0, 2)
        p = p.reshape(-1, num_pairs, self.embed_dim)
        q = q.reshape(-1, num_pairs, self.embed_dim)

        ip = torch.sum(p * q, -1).reshape(-1, num_pairs)
        l = torch.cat([xw, ip], 1)

        h = self.act(self.linear1(l))

        h = self.dropout(h)
        p = self.linear2(h).reshape(-1)

        return h, p

