# ----------------------------------------------------------------------
#
# File: utils.py
#
# Last edited: 19.04.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
# - Philip Wiese, pwiese@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
 

import torch
import numpy as np
from typing import Literal
from collections.abc import Iterable
from torch.fx.graph_module import GraphModule
from functools import reduce


class OutputProjectionITA(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, heads: int, weights: torch.Tensor = None, bias: torch.Tensor = None):
        super().__init__()

        self.heads = heads
        self.head_linear = torch.nn.ModuleList([
            torch.nn.Linear(in_features // heads, out_features, bias=True)] + 
            [torch.nn.Linear(in_features // heads, out_features, bias=False) for head in range(heads-1)])
        
        if weights is not None and bias is not None:
            sliced_weights = torch.split(weights, in_features // heads, dim=1)
            for i, linear in enumerate(self.head_linear):
                linear.weight.data = sliced_weights[i]
            self.head_linear[0].bias.data = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [batch, seq_len, embed_dim]
        x = x.reshape((x.shape[0], x.shape[1], self.heads, x.shape[2] // self.heads))
        x = x.transpose(1, 2)
        head_outputs = []
        for i in range(self.heads):
            head_outputs.append(self.head_linear[i](x[:, i, :, :]))
        output = reduce(torch.add, head_outputs)
        return output
    

# JUNGVI: Test the ITA output projection layer
if __name__ == "__main__":

    goldenModel = torch.nn.Linear(1024, 128, bias=True)
    modelITA = OutputProjectionITA(1024, 128, 4, weights=goldenModel.weight, bias=goldenModel.bias)

    inp_ita = torch.randn(1, 4, 5, 256)
    inp_golden = inp_ita.transpose(1, 2)
    inp_golden = inp_golden.reshape((1, 5, 1024))

    out_golden = goldenModel(inp_golden)
    out_ita = modelITA(inp_golden)
    mae = torch.mean(torch.abs(out_golden - out_ita)) 
    import IPython; IPython.embed()


def print_tabular(gm: GraphModule):
    """
        Prints the intermediate representation of the graph in tabular
        format with quanitzation metadata.
        """
    try:
        from tabulate import tabulate
    except ImportError:
        print("`print_tabular` relies on the library `tabulate`, "
              "which could not be found on this machine. Run `pip "
              "install tabulate` to install the library.")
        raise

    def quant_info(node, prop: Literal['eps', 'n_levels', 'signed'] = 'eps'):
        if 'quant' in node.meta:
            _repr = ''
            if prop == 'eps':
                _repr += str(node.meta['quant'].eps_in)
                _repr += ' -> '
                _repr += str(node.meta['quant'].eps_out)
            elif prop == 'n_levels':
                _repr += str(np.ceil(np.log2(node.meta['quant'].n_levels_in)).astype(int))
                _repr += ' -> '
                _repr += str(np.ceil(np.log2(node.meta['quant'].n_levels_out)).astype(int))
            elif prop == 'signed':
                _repr += str(node.meta['quant'].signed_in)
                _repr += ' -> '
                _repr += str(node.meta['quant'].signed_out)
            return _repr
        else:
            return '{}'

    def class_info(node):
        if node.op == 'call_module':
            return gm.get_submodule(node.target).__class__.__name__
        else:
            return ''

    node_specs = [[
        n.op,
        class_info(n), n.name, n.target,
        quant_info(n, 'n_levels'),
        quant_info(n, 'signed')
    ] for n in gm.graph.nodes]
    print(tabulate(node_specs, headers = ['opcode', 'class', 'name', 'target', 'n_levels', 'signed']))



def _getAbsMinAbsMax(tensor, n_levels=256):
    if tensor.numel() == 0:
        return torch.tensor(0), torch.tensor(1)

    _max = tensor.max()
    _min = tensor.min()

    if _max == 0 and _min == 0:
        _max = torch.tensor(1)

    absMax = torch.max(_max, torch.abs(_min))

    if _min == 0:
        absMin = torch.tensor(0)
    else:
        absMin = -absMax / ((n_levels // 2) - 1) * (n_levels // 2)

    return absMin.type_as(tensor), absMax.type_as(tensor)


def _getAdhocEpsList(n_levels: int = 256, *inputTensors):

    epsList = []

    for tensor in inputTensors:

        absMin, absMax = _getAbsMinAbsMax(tensor)

        eps = (absMax - absMin) / (n_levels - 1)
        epsList.append(eps)

    return epsList


def roundTensors(fakeBatch, eps_in):

    allRounded = []

    for tensor, eps in zip(fakeBatch, eps_in):
        # Skip if tensor is empty
        if tensor.numel() == 0:
            allRounded.append(tensor)
            continue

        absMin, absMax = _getAbsMinAbsMax(tensor)

        eps = eps.type_as(tensor)

        rounded = torch.trunc(torch.clamp(tensor, min=absMin, max=absMax) / eps) * eps
        allRounded.append(rounded)

    return allRounded


def delistify(_list):
    retList = []
    for arg in _list:
        if isinstance(arg, Iterable) and not isinstance(arg, torch.Tensor):
            retList += delistify(arg)
        else:
            retList.append(arg)
    return retList

def getMAEPercent(a: torch.Tensor, b: torch.Tensor):
        return 100 * torch.mean(torch.abs(a - b)) / max(torch.max(a).abs() + torch.min(a).abs(), torch.max(b).abs() + torch.min(b).abs())