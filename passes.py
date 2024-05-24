from typing import Union, Optional, Tuple, List, Literal
from dataclasses import dataclass
from functools import partial

from quantlib.editing.fx.passes.eps import QuantInfo
import numpy as np

import torch
from torch import fx, nn
from torch.fx.subgraph_rewriter import Match
from softermax import SofterMax

import quantlib.algorithms as qla
from quantlib.algorithms.pact.pact_ops import *

from quantlib.editing.fx.passes.pact import FxPass, ReplaceSequentialPatternPass, SequentialPass
from quantlib.editing.fx.passes import extract_eps
from quantlib.editing.fx.util import gm_modules, module_of_node

from quantlib.editing.fx.passes.pact.pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACTTracer, PACT_symbolic_trace, PACT_symbolic_trace_inclusive


class Observer(nn.Module):
    """
    Identity, model that can be used for debugging.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x


class NormalizeSoftmaxOutput(nn.Module):
    """
    A module that normalizes integer softmax outputs to a probability distribution.
    """

    def __init__(self, n_levels, dim = -1):
        super().__init__()
        self.dim = dim
        self.n_levels = n_levels

    def forward(self, x):
        # Convert integer output to float for division operation
        x_float = x.float()

        normalized_output = x_float/(self.n_levels-1.)
        # Calculate the sum of the softmax outputs across the specified dimension
        # sum_x = torch.sum(x_float, dim=self.dim, keepdim=True)

        # # Normalize each element by the sum to get probabilities that sum to 1
        # normalized_output = x_float / sum_x

        return normalized_output


def integerize_softmax_fun(gm: fx.GraphModule,
                           match: Match,
                           mode: Literal["I-BERT", "ITA",
                                         'ITA-Partial'] = "I-BERT",
                           D=2**24,
                           export_node=False):
    modules = gm_modules(gm)
    matched_nodes = [
        m for k, m in match.nodes_map.items() if k.op == 'call_module'
    ]
    lin_node = matched_nodes[0]
    matched_modules = [
        modules[m.target] for k, m in match.nodes_map.items()
        if k.op == 'call_module'
    ][::-1]
    module = matched_modules[0]
    eps_in = extract_eps(lin_node.meta['quant'].eps_in)
    # assert isinstance(module, PACTSoftmax), f"integerize_softmax_fun got bad match - expected PACTSoftmax, got {type(module)}"

    if mode == 'I-BERT':
        new_softmax = nn.Sequential(PACTIntegerSoftmax(n_levels=module.n_levels,
                                         eps_in=eps_in,
                                         export_node=export_node), NormalizeSoftmaxOutput(n_levels=module.n_levels))
    elif mode == 'ITA':
        new_softmax = nn.Sequential(PACTIntegerITAMax(max_value=module.act.max,
                                        n_levels=module.n_levels,
                                        eps_in=eps_in,
                                        D=D,
                                        export_node=export_node), NormalizeSoftmaxOutput(n_levels=module.n_levels))
    elif mode == 'ITA-Partial':
        new_softmax = nn.Sequential(PACTIntegerITAPartialMax(max_value=module.act.max,
                                               n_levels=module.n_levels,
                                               eps_in=eps_in,
                                               D=D,
                                               export_node=export_node), NormalizeSoftmaxOutput(n_levels=module.n_levels))
    else:
        assert False, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"

    return new_softmax

class IntegerizeSoftmaxPass(SequentialPass):
    def __init__(self, D=2**24, export_softmax_node = False , **kwargs):
        passes = []

        pattern = nn.Sequential(PACTSoftmax())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(integerize_softmax_fun, mode='I-BERT', export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))

        pattern = nn.Sequential(PACTITAMax())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(integerize_softmax_fun, mode='ITA', D=D, export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))

        pattern = nn.Sequential(PACTITAPartialMax())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(integerize_softmax_fun, mode='ITA-Partial', D=D, export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_SOFTMAX_PASS')


def replSoftmax(gm: fx.GraphModule, match: Match, mode: str, **kwargs):

    if 'n_levels' in kwargs:
        n_levels = kwargs['n_levels']
    else:
        n_levels = 2**8
    if mode == "I-BERT":
        replacement_class = nn.Sequential(PACTAsymmetricAct(**kwargs), PACTSoftmax(n_levels))
    elif mode == 'ITA':
        replacement_class = nn.Sequential(PACTITAMax(n_levels))
    elif mode == 'ITA-Partial':
        replacement_class = nn.Sequential(PACTITAPartialMax(n_levels=n_levels))

    return replacement_class


class ApproximateSoftmaxPass(SequentialPass):

    modes = ["I-BERT", "ITA", 'ITA-Partial']

    def __init__(self, mode: Literal["I-BERT", "ITA", 'ITA-Partial'] = "I-BERT", **kwargs):
        passes = []
        pattern = nn.Sequential(nn.Softmax())
        assert mode in self.modes, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(replSoftmax, mode=mode, **kwargs), f'_APPROXIMATE_SOFTMAX_PASS'))
        super().__init__(*passes, name_prefix='_APPROXIMATE_SOFTMAX_PASS')


class CustomAnnotateEpsPass(FxPass):

    def __init__(self, verbose = False):
        self.verbose = verbose
        super(CustomAnnotateEpsPass, self).__init__()

    def run_pass(self, gm: fx.GraphModule):
        if self.verbose:
            print(f"=> Running CustomAnnotateEpsPass")
        for node in gm.graph.nodes:
            eps_in = None
            eps_out = None

            if node.op == 'call_module':
                m = module_of_node(gm, node)

                if (isinstance(m, PACTSoftmax) or isinstance(m, PACTITAMax) or isinstance(m, PACTITAPartialMax)):
                    eps_in = [[i.meta['quant'].eps_out for i in node.args if isinstance(i, fx.Node)], {}]
                    if isinstance(m, PACTITAMax) or isinstance(m, PACTITAPartialMax):
                        eps_in = [[torch.Tensor((1.0,))], {}]
                    eps_out = torch.Tensor((1. / (m.n_levels - 1.),))
                    if self.verbose:
                        print(f" - {node.name:<40} ({m.__class__.__name__:<20}) {eps_in} -> {eps_out}")
                elif (isinstance(m, qla.pact._PACTActivation)):
                    eps_in = [[torch.Tensor((1.0,))], {}]
                    eps_out = m.get_eps()
                    if self.verbose:
                        print(f" - {node.name:<40} ({m.__class__.__name__:<20}) {eps_in} -> {eps_out}")

            node.meta['quant'] = QuantInfo(eps_in = eps_in,
                                           eps_out = eps_out,
                                           n_levels_in = None,
                                           n_levels_out = None,
                                           signed_in = None,
                                           signed_out = None)

        return gm
