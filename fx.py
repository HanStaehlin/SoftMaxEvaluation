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
import numpy as np
from scipy.stats import gaussian_kde

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

    def __init__(self, mod, plotter):
        self.mod = mod
        self.graph = mod.graph
        self.modules = dict(self.mod.named_modules())
        self.env = {}
        self.plotter = plotter

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
        count = 0
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
                    m.updateClipBounds()
                    self.plotter.add_histogram(m.histogram.clone(), m.clip_lo, m.clip_hi, count, epoch, m.truemin.clone(), m.truemax.clone(),
                                        m.running_mean, m.max)
                    count = count + 1
                    if "23" in node.name:
                        self.plotter.save_histograms()
                if isinstance(m, qla.pact.PACTITAMax) or isinstance(m, qla.pact.PACTITAPartialMax):
                    m=m.act
                    m.updateClipBounds()
                    self.plotter.add_histogram(m.histogram.clone(), m.clip_lo, m.clip_hi, count, epoch, m.truemin.clone(), m.truemax.clone(),
                                        m.running_mean, m.max)
                    count = count + 1
                    if "23" in node.name:
                        self.plotter.save_histograms()
            self.env[node.name] = result
        return result

import os
import torch
import matplotlib.pyplot as plt



class HistogramPlotter:
    def __init__(self, mode='individual', histogram={}):
        self.epoch_histograms = histogram
        self.mode = mode
        self.colors = ['blue', 'red', 'pink', 'green', 'purple', 'brown', 'orange', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black']
        self.all_node_names = []

    def add_histogram(self, histogram, clip_lo, clip_hi, node_name, epoch, truemin, truemax, running_mean, max):
        if epoch not in self.epoch_histograms:
            self.epoch_histograms[epoch] = []
        
        self.epoch_histograms[epoch].append({
            'histogram': histogram,
            'clip_lo': clip_lo,
            'clip_hi': clip_hi,
            'node_name': node_name,
            'truemin': truemin,
            'truemax': truemax,
            'running_mean': running_mean,
            'max': max
        })

        # Preserve the order of node names
        if node_name not in self.all_node_names:
            self.all_node_names.append(node_name)

    def save_histograms(self):
        if self.mode == 'individual':
            self._save_individual_histograms()
        elif self.mode == 'combined':
            self._save_combined_histograms()
        elif self.mode == 'both':
            self._save_individual_histograms()
            self._save_combined_histograms()

    def _save_individual_histograms(self):
        for epoch, histograms in self.epoch_histograms.items():
            os.makedirs(f"./histograms/{epoch}", exist_ok=True)
            num_histograms = len(histograms)
            cols = 4
            rows = (num_histograms + 1) // cols

            fig, axs = plt.subplots(rows, cols, figsize=(cols * 12, rows * 8))

            # Ensure axs is always iterable
            if num_histograms == 0:
                axs = [axs]
            else:
                axs = axs.flatten()

            for ax, hist_data in zip(axs, histograms):
                histogram = hist_data['histogram']
                clip_lo = hist_data['clip_lo'].item()
                clip_hi = hist_data['clip_hi'].item()
                node_name = hist_data['node_name']
                truemin = hist_data['truemin'].item()
                truemax = hist_data['truemax'].item()
                running_mean = hist_data['running_mean'].item()
                max_val = hist_data['max'].item()
                color = self.colors[epoch % len(self.colors)]
                bin_edges = torch.linspace(truemin, truemax, len(histogram) + 1)
                bar_width = max((truemin - truemin) / len(histogram), 0.2)

                if isinstance(histogram, torch.Tensor):
                    histogram = histogram.cpu().numpy()
                ax.bar(bin_edges[:-1].numpy(),
                    histogram,
                    width=bar_width,
                    align='edge',
                    color=color)
                if epoch < 10:
                    ax.axvline(x=clip_lo, color='red', linestyle='--', linewidth=3, alpha=0.5)
                    ax.axvline(x=clip_hi, color='green', linestyle='--', linewidth=3, alpha=0.5)
                # ax.axvline(x=running_mean, color='black', linestyle='--', linewidth=3, alpha=0.5)
                # ax.set_title(f"Node {node_name}", fontsize=24)
                ax.set_xlabel('Input Values', fontsize=28)
                ax.set_ylabel('Counts', fontsize=28)
                # ax.legend(fontsize=16)
                ax.tick_params(axis='both', which='major', labelsize=28)

            # Remove any extra subplots
            for ax in axs[len(histograms):]:
                fig.delaxes(ax)

            plt.tight_layout()
            plt.suptitle(f"Histograms for Epoch {epoch}", y=1.02, fontsize=28)
            plt.savefig(f"./histograms/{epoch}/epoch_{epoch}_histograms.png")
            plt.close()

    def _save_combined_density_plot(self, epoch, histograms):
        combined_data = []

        for hist_data in histograms:
            histogram = hist_data['histogram']
            truemin = hist_data['truemin'].item()
            truemax = hist_data['truemax'].item()
            bin_edges = torch.linspace(truemin, truemax, len(histogram) + 1)
            
            if isinstance(histogram, torch.Tensor):
                histogram = histogram.cpu().numpy()
            
            for i in range(len(histogram)):
                combined_data.extend([bin_edges[i].item()] * int(histogram[i]))

        if len(combined_data) == 0:
            print(f"No data for epoch {epoch} to generate combined density plot.")
            return

        combined_data = np.array(combined_data)
        density = gaussian_kde(combined_data)
        xs = np.linspace(combined_data.min(), combined_data.max(), 1000)
        density.covariance_factor = lambda: 0.25
        density._compute_covariance()
        
        plt.figure(figsize=(12, 8))
        plt.plot(xs, density(xs), color='blue', linewidth=2)
        plt.title(f"Combined Density Plot for Epoch {epoch}", fontsize=24)
        plt.xlabel('Input Values', fontsize=20)
        plt.ylabel('Density', fontsize=20)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"./histograms/{epoch}/epoch_{epoch}_combined_density.png")
        plt.close()

        print(f"Combined density plot for epoch {epoch} has been saved.")

    def _save_combined_histograms(self):
        os.makedirs(f"./histograms/combined", exist_ok=True)
        num_nodes = len(self.all_node_names)
        cols = 4
        rows = (num_nodes + 1) // cols
        fig, axs = plt.subplots(rows, cols, figsize=(cols*12, rows * 8))
        
        # Ensure axs is always iterable
        axs = axs.flatten()

        for node_name in self.all_node_names:
            ax = axs[self.all_node_names.index(node_name)]
            ax.set_title(f"Node {node_name}", fontsize=24)
            ax.set_xlabel('Input Values', fontsize=20)
            ax.set_ylabel('Distribution', fontsize=20)
            
            for epoch, histograms in self.epoch_histograms.items():
                hist_data = next((h for h in histograms if h['node_name'] == node_name), None)
                if hist_data:
                    print("Histogram epoch: ", epoch)
                    histogram = hist_data['histogram']
                    clip_lo = hist_data['clip_lo'].item()
                    clip_hi = hist_data['clip_hi'].item()
                    truemin = hist_data['truemin'].item()
                    truemax = hist_data['truemax'].item()
                    running_mean = hist_data['running_mean'].item()
                    max_val = hist_data['max'].item()
                    color = self.colors[epoch % len(self.colors)]
                    if isinstance(histogram, torch.Tensor):
                        histogram = histogram.cpu().numpy()
                    # Normalize the histogram
                    sum_hist_value = histogram.sum()
                    if sum_hist_value > 0:
                        histogram = histogram / sum_hist_value

                    # Set number of bins to 256 for epochs other than 0
                    num_bins = len(histogram)
                    bin_edges = torch.linspace(truemin, truemax, num_bins + 1)
                    bar_width = max((truemin - truemax) / (num_bins*2), 0.4)  # Minimum bar width set to 0.4
                    if epoch == 0:
                        label = 'Original'
                    elif epoch == 1:
                        label = 'Fake Quantized'
                    elif epoch == 10:
                        label = 'True Quantized'
                    else:
                        label = f'Epoch {epoch}'

                    if epoch != 0:
                        ax.bar(bin_edges[:-1].numpy(),
                            histogram,
                            width=bar_width,
                            align='edge',
                            color=color,
                            alpha=0.5,
                            label=label)
                    else:
                        # Plot outline for epoch 0
                        ax.plot(bin_edges[:-1].numpy(),
                                histogram*2**4*0.8,
                                color=color,
                                linewidth=5,
                                alpha=0.5,
                                label=label)
                        # ax.axvline(x=clip_lo, color='red', linestyle='--', linewidth=3, alpha=0.5)
                        # ax.axvline(x=clip_hi, color='green', linestyle='--', linewidth=3, alpha=0.5)
                        # ax.axvline(x=running_mean, color='black', linestyle='--', linewidth=3, alpha=0.5)

            ax.tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        plt.suptitle("Combined Histograms", fontsize=28)
        plt.savefig(f"./histograms/combined/combined_histograms.png")
        plt.close()


