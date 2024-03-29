# ----------------------------------------------------------------------
#
# File: main.py
#
# Last edited: 25.03.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Hannes Stählin (hstaehlin@ethz.ch), ETH Zurich
#         Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
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
import matplotlib.pyplot as plt

def util_main(**kwargs):
    

    def calculate_differences(N_range):
        B = 8
        log2e = np.log2(np.exp(1))
        eps_max = B / (2**B)

        l2_differences = []
        l1_differences = []
        partial_l2_differences = []
        partial_l1_differences = []
        integer_l2_differences = []
        integer_l1_differences = []
        fast_integer_l2_differences = []  # Added line
        fast_integer_l1_differences = []  # Added line
        partial_integer_l2_differences = []  # Added line
        partial_integer_l1_differences = []  # Added line

        for N in N_range:
            A = np.random.randint(-128, 127, size=(1, N, N), dtype=np.int8)
            input_float = A * eps_max  # Assume eps is eps_max
            input_int = A

            fast_softmax = fastSoftmax(input_float, False)
            fast_integer_softmax = fastSoftmax(input_int, True) / 255

            fast_partial_softmax = streamingPartialSoftmax(input_float, False)
            fast_partial_integer_softmax = streamingPartialSoftmax(input_int, True) / 255

            softmax = realSoftmax(input_float, False)
            integer_softmax = realSoftmax(input_int, True) / 255

            l2_diff = np.linalg.norm((softmax - fast_softmax)[0], 2)
            l1_diff = np.linalg.norm((softmax - fast_softmax)[0], 1)
            partial_l2_diff = np.linalg.norm((softmax - fast_partial_softmax)[0], 2)
            partial_l1_diff = np.linalg.norm((softmax - fast_partial_softmax)[0], 1)
            integer_l2_diff = np.linalg.norm((softmax - integer_softmax)[0], 2)
            integer_l1_diff = np.linalg.norm((softmax - integer_softmax)[0], 1)
            fast_integer_l2_diff = np.linalg.norm((softmax - fast_integer_softmax)[0], 2)  # Added line
            fast_integer_l1_diff = np.linalg.norm((softmax - fast_integer_softmax)[0], 1)  # Added line
            partial_integer_l2_diff = np.linalg.norm((softmax - fast_partial_integer_softmax)[0], 2)  # Added line
            partial_integer_l1_diff = np.linalg.norm((softmax - fast_partial_integer_softmax)[0], 1)  # Added line

            l2_differences.append(l2_diff)
            l1_differences.append(l1_diff)
            partial_l2_differences.append(partial_l2_diff)
            partial_l1_differences.append(partial_l1_diff)
            integer_l2_differences.append(integer_l2_diff)
            integer_l1_differences.append(integer_l1_diff)
            fast_integer_l2_differences.append(fast_integer_l2_diff)  # Added line
            fast_integer_l1_differences.append(fast_integer_l1_diff)  # Added line
            partial_integer_l2_differences.append(partial_integer_l2_diff)  # Added line
            partial_integer_l1_differences.append(partial_integer_l1_diff)  # Added line

        return l2_differences, l1_differences, partial_l2_differences, partial_l1_differences, integer_l2_differences, integer_l1_differences, fast_integer_l2_differences, fast_integer_l1_differences, partial_integer_l2_differences, partial_integer_l1_differences  # Modified line

    # Rest of the code remains the same

    N_range = [64, 128, 256, 512, 1024]
    l2_differences, l1_differences, partial_l2_differences, partial_l1_differences, integer_l2_differences, integer_l1_differences, fast_integer_l2_differences, fast_integer_l1_differences, partial_integer_l2_differences, partial_integer_l1_differences = calculate_differences(N_range)

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(N_range, l2_differences, label='L2 Differences (Fast Softmax)')
    plt.plot(N_range, partial_l2_differences, label='L2 Differences (Partial Softmax)')
    plt.xlabel('N')
    plt.ylabel('L2 Differences')
    plt.title('Softmax L2 Differences for Different N (Non-Integer Softmax)')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(N_range, integer_l2_differences, label='L2 Differences (Integer Softmax)')
    plt.plot(N_range, fast_integer_l2_differences, label='L2 Differences (Fast Integer Softmax)')
    plt.plot(N_range, partial_integer_l2_differences, label='L2 Differences (Partial Integer Softmax)')
    plt.xlabel('N')
    plt.ylabel('L2 Differences')
    plt.title('Softmax L2 Differences for Different N (Integer Softmax)')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(N_range, l1_differences, label='L1 Differences (Fast Softmax)')
    plt.plot(N_range, partial_l1_differences, label='L1 Differences (Partial Softmax)')
    plt.xlabel('N')
    plt.ylabel('L1 Differences')
    plt.title('Softmax L1 Differences for Different N (Non-Integer Softmax)')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(N_range, integer_l1_differences, label='L1 Differences (Integer Softmax)')
    plt.plot(N_range, fast_integer_l1_differences, label='L1 Differences (Fast Integer Softmax)')
    plt.plot(N_range, partial_integer_l1_differences, label='L1 Differences (Partial Integer Softmax)')
    plt.xlabel('N')
    plt.ylabel('L1 Differences')
    plt.title('Softmax L1 Differences for Different N (Integer Softmax)')
    plt.legend()

    plt.tight_layout()
    plt.savefig("plot.png")


util_main()
