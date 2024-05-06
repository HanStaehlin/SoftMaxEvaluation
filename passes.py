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


class ApproximateSoftmaxPass(SequentialPass):
    def __init__(self, mode: Literal["I-BERT", "ITA", 'ITA-Partial'] = "I-BERT", **kwargs):
        passes = []
        pattern = nn.Sequential(nn.Softmax())

        if mode=='I-BERT':
            replacement_class = nn.Sequential(PACTUnsignedAct(**kwargs),PACTSoftmax())
        elif mode=='ITA':
            replacement_class = nn.Sequential(PACTUnsignedAct(**kwargs),PACTITAMax())
        elif mode=='ITA-Partial':
            replacement_class = PACTITAPartialMax()
        else:
            assert False, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"
    
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, lambda x,y: replacement_class, f'_APPROXIMATE_SOFTMAX_PASS'))

        super().__init__(*passes, name_prefix='_APPROXIMATE_SOFTMAX_PASS')