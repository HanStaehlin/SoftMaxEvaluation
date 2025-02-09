import torch
import math
import numpy as np
import torch.nn as nn

#Not Functional
class SofterMax(nn.Module):

    def __init__(self, version=1):
        super(SofterMax, self).__init__()
        self.version = version

    def forward(self, x):
        if self.version == 1:
            return self.softermax_v1(x)
        elif self.version == 2:
            return self.softermax_v2(x)
        elif self.version == 3:
            return self.softermax_v3(x)
        else:
            raise ValueError("Unsupported Softermax version")

    def softermax_v1(self, x):
        V = x.shape[-1]
        m = -float('inf') * torch.ones(
            (x.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        m.fill_(-float('inf'))
        d = torch.zeros_like(x[:, :, 0])

        for j in range(V):
            mj = torch.max(m, x[:, :, j])
            d += 2**(x[:, :, j] - mj)
            m = mj

        y = torch.zeros_like(x)
        for i in range(V):
            y[:, :, i] = 2**(x[:, :, i] - m) / d

        return y

    def softermax_v2(self, x):
        V = x.shape[-1]
        m0 = torch.full((x.shape[0], x.shape[1]),
                        -float('inf'),
                        device=x.device)
        d = torch.zeros_like(x[:, :, 0])

        for j in range(V):
            mj = torch.max(m0, x[:, :, j])
            d = d * 2**(mj - m0) + 2**(x[:, :, j] - mj)
            m0 = mj

        y = torch.zeros_like(x)
        for i in range(V):
            y[:, :, i] = 2**(x[:, :, i] - m0) / d

        return y

    def softermax_v3(self, x):
        V = x.shape[-1]
        m0 = torch.full((x.shape[0], x.shape[1]),
                        -float('inf'),
                        device=x.device,
                        dtype=x.dtype)
        d = torch.zeros_like(x[:, :, 0])

        for j in range(V):
            mj = torch.max(m0, x[:, :, j])
            d = d >> (mj - m0) + 2**(x[:, :, j] - mj)
            m0 = mj

        y = torch.zeros_like(x)
        for i in range(V):
            y[:, :, i] = (2**(x[:, :, i] - m0) >> (m0 - x[:, :, i])) / d

        return y

# class SofterMax(torch.nn.Module):

#     def __init__(self, n_bits: int = 256, integerize: bool = False):
#         super().__init__()
#         self.n_bits = n_bits
#         self.integerize = integerize
#         self.log2e = math.log2(math.exp(1))
#         self.eps_x = n_bits / (2**n_bits * self.log2e)

#     def forward(self, A_requant):
#         # Get the number of heads
#         n_heads = A_requant.shape[-3]

#         # Scale the input values if integerize is True
#         if self.integerize:
#             x = A_requant * self.eps_x
#         else:
#             x = A_requant.float()
#         x = x[0]
#         max_values = -np.inf * np.ones((n_heads, x.shape[1], x.shape[2]))
#         exp_result = np.zeros((n_heads, x.shape[1], x.shape[2]))
#         sum = 0

#         # Calculate the exponential values
#         # Loop over each head
#         for i in range(n_heads):
#             # Loop over each row
#             for j in range(x.shape[1]):
#                 # Loop over each element in the row
#                 max_val = x[i, j, 0]
#                 for k in range(x.shape[2]):
#                     # Calculate the exponentiation
#                     if x[i, j, k] > max_val:
#                         max_val = x[i, j, k]
#                     max_values[i, j, k] = int(max_val)
#                     if (k == 0):
#                         exp_result[i, j, k] = 1
#                         sum = exp_result[i, j, k]
#                     else:
#                         exp_result[i, j,
#                                    k] = 2**(x[i, j, k] - max_values[i, j, k])
#                         sum = int(exp_result[i, j, k] +
#                                   sum) >> int(max_values[i, j, k] -
#                                               max_values[i, j, k - 1])

#         #Normalize the exponential values to get the softmax probabilities
#         # Loop over each head
#         for i in range(n_heads):
#             # Loop over each row
#             for j in range(x.shape[1]):
#                 # Loop over each element in the row
#                 for k in range(x.shape[2]):
#                     exp_result[i, j, k] = (int(exp_result[i, j, k]) >> int(
#                         max_values[i, j, -1] - max_values[i, j, k])) / sum

#         A_requant[0] = torch.tensor(exp_result)
#         # Normalize the exponential values to get the softmax probabilities
#         if self.integerize:
#             # Scale the probabilities to the range [0, 2^7 - 1]
#             return A_requant * (2**7 - 1)
#         else:
#             return A_requant

# class SofterMaxInteger(torch.nn.Module):
#     def __init__(self, n_bits: int = 256, n_levels: int = 128):
#         super().__init__()
#         self.n_bits = n_bits
#         self.n_levels = n_levels
#         self.log2e = math.log2(math.exp(1))
#         self.eps_x = n_bits / (2**n_bits * self.log2e)

#     def forward(self, A_requant):
#         x = A_requant * self.eps_x
#         n_heads = x.size(0)
#         max_values = torch.full_like(x, -float('inf'), dtype=torch.int32)
#         exp_result = torch.zeros_like(x, dtype=torch.int32)
#         softmax_result = torch.zeros_like(x, dtype=torch.float32)

#         for i in range(n_heads):
#             max_values[i] = torch.max(x[i], dim=-1, keepdim=True)[0].int()
#             exp_result[i] = (2 ** (x[i] - max_values[i])).int()
#             sum_exp = torch.sum(exp_result[i], dim=-1, keepdim=True).int()
#             softmax_result[i] = (exp_result[i] / sum_exp).float() * (self.n_levels - 1)

#         return softmax_result
