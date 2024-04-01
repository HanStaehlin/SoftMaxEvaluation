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
from softmax_golden import fastSoftmax, realSoftmax, streamingPartialSoftmax, softermaxStep1, softermaxStep2
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
        fast_integer_l2_differences = []  
        fast_integer_l1_differences = []  
        partial_integer_l2_differences = []  
        partial_integer_l1_differences = []  

        softermax_step1_l2_differences = []
        softermax_step1_l1_differences = []

        softermax_step2_l2_differences = []
        softermax_step2_l1_differences = []

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

            softermax_step1 = softermaxStep1(input_float, False)
            softermax_step2 = softermaxStep2(input_float, False)

            l2_diff = np.linalg.norm((softmax - fast_softmax)[0], 2)
            l1_diff = np.linalg.norm((softmax - fast_softmax)[0], 1)
            partial_l2_diff = np.linalg.norm((softmax - fast_partial_softmax)[0], 2)
            partial_l1_diff = np.linalg.norm((softmax - fast_partial_softmax)[0], 1)
            integer_l2_diff = np.linalg.norm((softmax - integer_softmax)[0], 2)
            integer_l1_diff = np.linalg.norm((softmax - integer_softmax)[0], 1)
            fast_integer_l2_diff = np.linalg.norm((softmax - fast_integer_softmax)[0], 2)  
            fast_integer_l1_diff = np.linalg.norm((softmax - fast_integer_softmax)[0], 1)  
            partial_integer_l2_diff = np.linalg.norm((softmax - fast_partial_integer_softmax)[0], 2)  
            partial_integer_l1_diff = np.linalg.norm((softmax - fast_partial_integer_softmax)[0], 1) 
            softermax_step1_l2_diff = np.linalg.norm((softmax - softermax_step1)[0], 2)
            softermax_step1_l1_diff = np.linalg.norm((softmax - softermax_step1)[0], 1)
            softermax_step2_l2_diff = np.linalg.norm((softmax - softermax_step2)[0], 2)
            softermax_step2_l1_diff = np.linalg.norm((softmax - softermax_step2)[0], 1)
            
            l2_differences.append(l2_diff)
            l1_differences.append(l1_diff)
            partial_l2_differences.append(partial_l2_diff)
            partial_l1_differences.append(partial_l1_diff)
            integer_l2_differences.append(integer_l2_diff)
            integer_l1_differences.append(integer_l1_diff)
            fast_integer_l2_differences.append(fast_integer_l2_diff)  
            fast_integer_l1_differences.append(fast_integer_l1_diff)  
            partial_integer_l2_differences.append(partial_integer_l2_diff)  
            partial_integer_l1_differences.append(partial_integer_l1_diff) 
            softermax_step1_l2_differences.append(softermax_step1_l2_diff)
            softermax_step1_l1_differences.append(softermax_step1_l1_diff)
            softermax_step2_l2_differences.append(softermax_step2_l2_diff)
            softermax_step2_l1_differences.append(softermax_step2_l1_diff)    

        return l2_differences, l1_differences, partial_l2_differences, partial_l1_differences, integer_l2_differences, integer_l1_differences, fast_integer_l2_differences, fast_integer_l1_differences, partial_integer_l2_differences, partial_integer_l1_differences, softermax_step1_l2_differences, softermax_step1_l1_differences, softermax_step2_l2_differences, softermax_step2_l1_differences

    N_range = [16,32,64, 128, 256, 512, 1024, 2048]
    l2_differences, l1_differences, partial_l2_differences, partial_l1_differences, integer_l2_differences, integer_l1_differences, fast_integer_l2_differences, fast_integer_l1_differences, partial_integer_l2_differences, partial_integer_l1_differences, softermax_step1_l2_differences, softermax_step1_l1_differences, softermax_step2_l2_differences, softermax_step2_l1_differences = calculate_differences(N_range)

    print("Fast L2 Differences:", l2_differences)
    print("Fast L1 Differences:", l1_differences)
    print("Partial L2 Differences:", partial_l2_differences)
    print("Partial L1 Differences:", partial_l1_differences)
    print("Integer L2 Differences:", integer_l2_differences)
    print("Integer L1 Differences:", integer_l1_differences)
    print("Fast Integer L2 Differences:", fast_integer_l2_differences)
    print("Fast Integer L1 Differences:", fast_integer_l1_differences)
    print("Partial Integer L2 Differences:", partial_integer_l2_differences)
    print("Partial Integer L1 Differences:", partial_integer_l1_differences)
    print("Softermax Step 1 L2 Differences:", softermax_step1_l2_differences)
    print("Softermax Step 1 L1 Differences:", softermax_step1_l1_differences)
    print("Softermax Step 2 L2 Differences:", softermax_step2_l2_differences)
    print("Softermax Step 2 L1 Differences:", softermax_step2_l1_differences)
    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.plot(N_range, l2_differences, label='L2 Differences (Fast Softmax)')
    plt.plot(N_range, partial_l2_differences, label='L2 Differences (Partial Softmax)')
    plt.plot(N_range, softermax_step1_l2_differences, label='L2 Differences (Softermax Step 1)')
    plt.plot(N_range, softermax_step2_l2_differences, label='L2 Differences (Softermax Step 2)')
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
    plt.plot(N_range, softermax_step1_l1_differences, label='L2 Differences (Softermax Step 1)')
    plt.plot(N_range, softermax_step2_l1_differences, label='L2 Differences (Softermax Step 2)')
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
