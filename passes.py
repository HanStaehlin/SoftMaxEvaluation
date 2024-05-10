from typing import Union, Optional, Tuple, List, Literal
from dataclasses import dataclass
from functools import partial

import numpy as np

import torch
from torch import fx, nn
from torch.fx.subgraph_rewriter import Match

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
    The model class, which defines our classifier.
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


class ApproximateSoftmaxPass(SequentialPass):

    def __init__(self,
                 mode: Literal["I-BERT", "ITA", 'ITA-Partial', 'II-BERT',
                               'IITA'] = "I-BERT",
                 **kwargs):
        passes = []
        pattern = nn.Sequential(nn.Softmax())

        if mode == 'I-BERT':
            replacement_class = nn.Sequential(Observer("before act"),
                                              PACTUnsignedAct(**kwargs),
                                              Observer(name="after act"),
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
        else:
            assert False, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"

        passes.append(
            ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace,
                                         lambda x, y: replacement_class,
                                         f'_APPROXIMATE_SOFTMAX_PASS'))

        super().__init__(*passes, name_prefix='_APPROXIMATE_SOFTMAX_PASS')
