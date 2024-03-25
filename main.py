# ----------------------------------------------------------------------
#
# File: main.py
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
import numpy as np
from softmax_golden import fastSoftmax, realSoftmax, streamingPartialSoftmax

def util_main(**kwargs):
    B = 8
    log2e = np.log2(np.exp(1))
    eps_max = B / (2**B)

    N = 1024
    A = np.random.randint(-128, 127, size = (1, N, N), dtype = np.int8)
    input_float = A * eps_max  # Assume eps is eps_max
    input_int = A

    fast_softmax = fastSoftmax(input_float, False)
    fast_integer_softmax = fastSoftmax(input_int, True) / 255

    fast_partial_softmax = streamingPartialSoftmax(input_float, False)
    fast_partial_integer_softmax = streamingPartialSoftmax(input_int, True) / 255

    softmax = realSoftmax(input_float, False)
    integer_softmax = realSoftmax(input_int, True) / 255

    print(f"=> L2 Softmax Differences:")
    print(
        f"  Softmax              - Fast Softmax                    : {np.linalg.norm((softmax-fast_softmax)[0], 2):.10}"
    )
    print(
        f"  Softmax              - Fast Partial Softmax            : {np.linalg.norm((softmax-fast_partial_softmax)[0], 2):.10}"
    )
    print(
        f"  Softmax              - Fast Integer Softmax            : {np.linalg.norm((softmax-fast_integer_softmax)[0], 2):.10}"
    )
    print(
        f"  Softmax              - Fast Partial Integer Softmax    : {np.linalg.norm((softmax-fast_partial_integer_softmax)[0], 2):.10}"
    )
util_main()