# ----------------------------------------------------------------------
#
# File: evaluations_quantlib.py
#
# Last edited: 25.03.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Hannes Stählin (hstaehlin@ethz.ch), ETH Zurich
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
import numpy as np
from quantlib.algorithms.pact.pact_ops import (PACTITAMax,
                                               PACTSoftmax,
                                               PACTIntegerSoftmax)


def run_softmax_comparisons(num_iterations=1):
  """
  Runs softmax comparisons with different random inputs and prints results.

  Args:
      num_iterations: Number of iterations to run the comparison (default: 10).
  """
  for _ in range(num_iterations):
    N = 1024

    # Generate different random input variants:
    input_variants = [
        # Uniform distribution 
        ["Random",torch.rand(1, N, N) * 255 - 128],
        # Normal distribution 
        ["Normal",torch.randn(1, N, N)],
        # Sparse inputs with most values near 0
        ["Sparse",torch.randint(-5, 5, size=(1, N, N), dtype=torch.float32) ],
    ]

    for input_data in input_variants:
      input = input_data[1].unsqueeze(0).float()

      ITAMax = PACTITAMax()
      Softmax = PACTSoftmax()
      IntegerSoftmax = PACTIntegerSoftmax()

      ITAmax_softmax = ITAMax.forward(input).detach().numpy().squeeze(axis=0)
      softmax = Softmax.forward(input).detach().numpy().squeeze(axis=0)
      integer_softmax = IntegerSoftmax.forward(input).detach().numpy().squeeze(axis=0)
      ITAMax.started = torch.tensor(1)

      log2e = np.log2(np.exp(1))
      eps = 8 / (2**8*log2e)
      ITAMax.set_eps_in(torch.tensor((eps, )))
      ITAMax_fake_quantized = ITAMax.forward(input).detach().numpy().squeeze(axis=0)

      print(f"\nInput Variant: {input_data[0]}")
      print(f"Softmax shape: {softmax.shape}")

      print(f" L2 Softmax Differences:")
      print(f"  Real Softmax              - ITAMax Softmax                  : {np.linalg.norm(softmax[0] - ITAmax_softmax[0], 2):.10}")
      print(f"  Real Softmax              - ITAMax Fake Quantized Softmax    : {np.linalg.norm(softmax[0] - ITAMax_fake_quantized[0], 2):.10}")
      print(f"  Real Softmax              - Integer Softmax                 : {np.linalg.norm(softmax[0] - integer_softmax[0], 2):.10}")


if __name__ == "__main__":
  run_softmax_comparisons()

