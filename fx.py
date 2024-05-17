# ----------------------------------------------------------------------
#
# File: fx.py
#
# Last edited: 21.02.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.fx
from torch.fx.node import Node

from transformers.utils.fx import HFTracer
from quantlib.editing.fx.passes.pact import PACT_OPS

import torch.nn as nn

from typing import Dict, List, Union
from quantlib.editing.fx.util import gm_modules, module_of_node
import quantlib.algorithms as qla
from quantlib.algorithms.pact.pact_ops import *
from matplotlib import pyplot as plt
import os

class HFLeafTracer(HFTracer):
    # Allows tracing modules with custom granularity: Any modules of a type
    # contained in the leaf_types list will not be traced through and will
    # instead be represented as call_module nodes.
    def __init__(self, leaf_types: List[torch.nn.Module] = list(PACT_OPS), *args, **kwargs):
        self.leaf_types = [] if leaf_types is None else leaf_types
        super().__init__(*args, **kwargs)

    def is_leaf_module(self, m : torch.nn.Module, module_qualified_name : str):
        base_condition = super(HFLeafTracer, self).is_leaf_module(m, module_qualified_name)

        return base_condition or isinstance(m, tuple(self.leaf_types))


class SimpleInterpreter:
    """
    Simple FX interpreter. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. A
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

        self.env: Dict[str, Node] = {}

    def load_arg(self, a):
        return torch.fx.graph.map_arg(a, lambda n: self.env[n.name])

    def fetch_attr(self, target: str):
        target_atoms = target.split('.')
        attr_itr = self.mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def propagate(self, *args, **kwargs):

        args_iter = iter(args)

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = self.fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*self.load_arg(node.args), **self.load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = self.load_arg(node.args)
                kwargs = self.load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*self.load_arg(node.args), **self.load_arg(node.kwargs))

            self.env[node.name] = result

        return result


class HistoryInterpreter:
    """
    Simple FX interpreter. This class takes a `GraphModule`.
    Then, its `propagate` method executes the `GraphModule`
    node-by-node with the given arguments. A
    """

    def __init__(self, n_epoch, n_batch, print_interval):
        self.mod: Union[None, torch.fx.GraphModule] = None
        self.graph: Union[None, torch.fx.Graph] = None
        self.modules: Union[None, Dict[str, nn.Module]] = None

        self.n_batch = n_batch
        self.n_epoch = n_epoch
        self.print_interval = print_interval

        self.envs: List[List[Dict[str, Node]]] = [[{} for _ in range(n_batch)] for _ in range(n_epoch)]

    def load_arg(self, a, n_epoch, n_batch):
        return torch.fx.graph.map_arg(a, lambda n: self.envs[n_epoch][n_batch][n.name])

    def fetch_attr(self, target: str):
        target_atoms = target.split('.')
        attr_itr = self.mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def propagate(self, mod: torch.fx.GraphModule, n_epoch, n_batch, *args, **kwargs):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())

        args_iter = iter(args)

        if (n_batch % self.print_interval == 0) and n_epoch >= 1:
            print(f"[HistoryInterpreter] Propagating epoch {n_epoch} batch {n_batch}")

        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = self.fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*self.load_arg(node.args, n_epoch, n_batch),
                                     **self.load_arg(node.kwargs, n_epoch, n_batch))
            elif node.op == 'call_method':
                self_obj, *args = self.load_arg(node.args, n_epoch, n_batch)
                kwargs = self.load_arg(node.kwargs, n_epoch, n_batch)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                result = self.modules[node.target](*self.load_arg(node.args, n_epoch, n_batch),
                                                   **self.load_arg(node.kwargs, n_epoch, n_batch))

            self.envs[n_epoch][n_batch][node.name] = result

            if (n_batch % self.print_interval == 0) and n_epoch >= 1 and isinstance(result, torch.Tensor):
                diff = torch.nn.MSELoss()(result, self.envs[0][n_batch][node.name])
                print(f"Node {node.name:<100}: {diff:.4e}")

        return result

class HistogramInterpreter:
    """
    Interpreter for FX graphs that logs and plots histogram data for specific module types.
    """

    def __init__(self, mod):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.env = {}

    def load_arg(self, a):
        return torch.fx.graph.map_arg(a, lambda n: self.env[n.name])

    def fetch_attr(self, target: str):
        target_atoms = target.split('.')
        attr_itr = self.mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    def propagate(self, epoch, *args, **kwargs):
        args_iter = iter(args)
        for node in self.graph.nodes:
            if node.op == 'placeholder':
                result = next(args_iter)
            elif node.op == 'get_attr':
                result = self.fetch_attr(node.target)
            elif node.op == 'call_function':
                result = node.target(*self.load_arg(node.args), **self.load_arg(node.kwargs))
            elif node.op == 'call_method':
                self_obj, *args = self.load_arg(node.args)
                kwargs = self.load_arg(node.kwargs)
                result = getattr(self_obj, node.target)(*args, **kwargs)
            elif node.op == 'call_module':
                m = self.modules[node.target]
                result = m(*self.load_arg(node.args), **self.load_arg(node.kwargs))
                if isinstance(m, qla.pact._PACTActivation):
                    self.plot_histogram(m.histogram, m.clip_lo, m.clip_hi, node.name, epoch, m.truemin, m.truemax)
            self.env[node.name] = result
        return result

    def plot_histogram(self, histogram, clip_lo, clip_hi, node_name, epoch, truemin, truemax):
        os.makedirs(f"./histograms/{epoch}", exist_ok=True)
        plt.figure()
        # Convert parameters to scalar values using .item()
        truemin = truemin.item()
        truemax = truemax.item()
        clip_lo = clip_lo.item()
        clip_hi = clip_hi.item()

        # Now use the scalar values in torch.linspace
        bin_edges = torch.linspace(truemin, truemax, len(histogram) + 1)

        plt.bar(bin_edges[:-1].numpy(), histogram.numpy(), width=(truemax-truemin)/len(histogram), align='edge', color='blue')
        plt.axvline(x=clip_lo, color='red', label='Low Clip')
        plt.axvline(x=clip_hi, color='green', label='High Clip')
        plt.title(f"Histogram for Node {node_name} Epoch {epoch}")
        plt.xlabel('Activation Values')
        plt.ylabel('Counts')
        plt.legend()
        plt.savefig(f"./histograms/{epoch}/{node_name}_histogram.png")
        plt.close()

