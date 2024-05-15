from typing import Union, Optional, Tuple, List, Literal
from dataclasses import dataclass
from functools import partial

import numpy as np

import torch
from torch import fx, nn
from torch.fx.subgraph_rewriter import Match
from softermax import SofterMax
from quantlib.algorithms.pact.pact_ops import *

from quantlib.editing.fx.passes.pact import FxPass, ReplaceSequentialPatternPass, ModifySequentialPatternPass, SequentialPass, ShapePropPass
from quantlib.editing.fx.passes.pact import AnnotateEpsPass, extract_eps
from quantlib.editing.fx.passes.pact import MergeConvBNPass, RetracePass
from quantlib.editing.fx.util import gm_modules, module_of_node
from quantlib.editing.fx.util.tracing import LeafTracer, custom_symbolic_trace

from quantlib.editing.fx.passes.pact.pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACTTracer, PACT_symbolic_trace, PACT_symbolic_trace_inclusive

quant_cfg = {
    "mul": torch.tensor(255),
    "add": torch.tensor(0),
    "n_levels": 2**16,
    "D": torch.tensor(1,),
}

dequant_cfg = {
    "mul": torch.tensor(1),
    "add": torch.tensor(0),
    "n_levels": 2**16,
    "D": torch.tensor(10,),
}


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
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # Convert integer output to float for division operation
        x_float = x.float()

        # Calculate the sum of the softmax outputs across the specified dimension
        sum_x = torch.sum(x_float, dim=self.dim, keepdim=True)

        # Normalize each element by the sum to get probabilities that sum to 1
        normalized_output = x_float / sum_x

        return normalized_output


def integerize_softmax_fun(gm: fx.GraphModule,
                           match: Match,
                           mode: Literal["I-BERT", "ITA",
                                         'ITA-Partial'] = "I-BERT",
                           D=2**12,
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
    eps_in = lin_node.meta['quant'].eps_in
    # assert isinstance(module, PACTSoftmax), f"integerize_softmax_fun got bad match - expected PACTSoftmax, got {type(module)}"

    if mode == 'I-BERT':
        new_softmax = nn.Sequential(PACTIntegerSoftmax(n_levels=module.n_levels,
                                         eps_in=eps_in,
                                         export_node=export_node), NormalizeSoftmaxOutput())
    elif mode == 'ITA':
        new_softmax = nn.Sequential(PACTIntegerITAMax(max_value=module.act.max,
                                        n_levels=module.n_levels,
                                        eps_in=eps_in,
                                        D=D,
                                        export_node=export_node), NormalizeSoftmaxOutput())
    elif mode == 'ITA-Partial':
        new_softmax = nn.Sequential(PACTIntegerITAPartialMax(max_value=module.act.max,
                                               n_levels=module.n_levels,
                                               eps_in=eps_in,
                                               D=D,
                                               export_node=export_node), NormalizeSoftmaxOutput())
    else:
        assert False, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"

    return new_softmax

class IntegerizeSoftmaxPass(SequentialPass):
    def __init__(self, D=2**12, export_softmax_node = False , **kwargs):
        passes = []

        pattern = nn.Sequential(PACTSoftmax())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(integerize_softmax_fun, mode='I-BERT', export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))

        pattern = nn.Sequential(PACTITAMax())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(integerize_softmax_fun, mode='ITA', D=D, export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))

        pattern = nn.Sequential(PACTITAPartialMax())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(integerize_softmax_fun, mode='ITA-Partial', D=D, export_node=export_softmax_node), f'_INTEGER_SOFTMAX_PASS'))
        super().__init__(*passes, name_prefix='_INTEGER_SOFTMAX_PASS')

class ApproximateSoftmaxPass(SequentialPass):

    def __init__(self,
                 mode: Literal["I-BERT", "ITA", 'ITA-Partial', 'II-BERT',
                               'IITA', 'SOFTER'] = "I-BERT",
                 **kwargs):
        passes = []
        pattern = nn.Sequential(nn.Softmax())

        if mode == 'I-BERT':
            replacement_class = nn.Sequential(PACTUnsignedAct(**kwargs),
                                              PACTSoftmax())
        elif mode == 'II-BERT':
            replacement_class = nn.Sequential(PACTUnsignedAct(**kwargs),
                                              Observer("before quant"),
                                              RequantShift(**quant_cfg),
                                              Observer(name="after quant"),
                                              PACTIntegerSoftmax(),
                                              Observer("after softmax"),
                                              NormalizeSoftmaxOutput())
        elif mode == 'ITA':
            replacement_class = nn.Sequential(PACTUnsignedAct(**kwargs),
                                              PACTITAMax())
        elif mode == 'IITA':
            replacement_class = nn.Sequential(PACTUnsignedAct(**kwargs),
                                              Observer("before quant"),
                                              RequantShift(**quant_cfg),
                                              Observer(name="after quant"),
                                              PACTIntegerITAMax(max_value=2**12), #TODO: set MaxValue
                                              Observer("after softmax"),
                                              NormalizeSoftmaxOutput())
        elif mode == 'ITA-Partial':
            replacement_class = PACTITAPartialMax()
        elif mode == 'SOFTER':
            replacement_class = nn.Sequential(SofterMax())
        else:
            assert False, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"

        passes.append(
            ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace,
                                         lambda x, y: replacement_class,
                                         f'_APPROXIMATE_SOFTMAX_PASS'))

        super().__init__(*passes, name_prefix='_APPROXIMATE_SOFTMAX_PASS')
