# ----------------------------------------------------------------------
#
# File: softmax.py
#
# Last edited: 5.03.2024
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

import numpy as np
import torch

def fastSoftmax(x, integerize = True):
    if not integerize:
        x = x.astype(np.float64)
    else:
        x = x.astype(np.int32)

    seq_length = x.shape[-1]
    n_heads = x.shape[-3]

    # Number of bits
    B = 8

    # Scaling factor
    eps_max = B / (2**B)

    # Find the maximum for each row in the current column block (consisting of 16 columns)
    max = np.repeat(np.max(x, axis = -1), seq_length).reshape(n_heads, seq_length, seq_length)

    # Find the difference between the maximum and x in the current part of the row
    diff = max - x

    # Shift the values by B-log2B -> multiply by B/2**B = log2e*eps_x
    # Make sure to do use round-half-up instead of round-half-to-even
    if integerize:
        shift = np.floor(diff * eps_max + 0.5 + np.finfo(np.float32).eps).astype(int)
    else:
        shift = diff

    # Calculate exponential sum over the  the row and scale it by 2**10 to prevent underflow
    if integerize:
        exp_sum = np.floor(np.sum(2**8 / 2**shift, axis = -1))
    else:
        exp_sum = np.sum(1 / 2**shift, axis = -1)

    # Invert the partial sum
    if integerize:
        exp_sum_inverse = np.floor((2**8 - 1) * 2**8 / exp_sum).astype(int)
    else:
        exp_sum_inverse = 1 / exp_sum

    # Calculate the activation value
    if integerize:
        return (np.floor(np.repeat(exp_sum_inverse, seq_length).reshape(n_heads, seq_length, seq_length) /
                         2**shift)).astype(np.int8)
    else:
        return np.repeat(exp_sum_inverse, seq_length).reshape(n_heads, seq_length, seq_length) / 2**shift


def streamingPartialSoftmax(x, integerize = True):
    if not integerize:
        x = x.astype(np.float32)

    seq_length = x.shape[-1]
    n_heads = x.shape[-3]
    width = 16  # 16 PE (processing units)
    groups = seq_length // width

    assert seq_length % width == 0, f"Sequence length must be a multiple of width ({width})"

    # Number of bits
    B = 8

    # Scaling factor
    eps_max = B / (2**B)

    if integerize:
        x = x
    else:
        x = x / eps_max

    # Initialize denominator
    if integerize:
        exp_partial_sum = np.zeros((n_heads, seq_length), dtype = np.int32)
    else:
        exp_partial_sum = np.zeros((n_heads, seq_length), dtype = np.float32)

    # Initialize maximum with minimal possible value
    # max = np.full((n_heads, seq_length), -2**(B - 1), dtype = np.int8)
    if integerize:
        global_max = np.full((n_heads, seq_length), -128, dtype = np.int8)
    else:
        global_max = np.full((n_heads, seq_length), -np.Infinity, dtype = np.float32)

    ## STAGE 1: Compute the denominator of the softmax
    for i in range(groups):
        # Find the maximum for each row in the current column block (consisting of 16 columns)
        if integerize:
            current_max = np.max(x[..., 0 + i * width:width + i * width].astype(np.int32), axis = -1)
        else:
            current_max = np.max(x[..., 0 + i * width:width + i * width].astype(np.float32), axis = -1)

        # Initialize all shift values for each row to zero
        if integerize:
            shift_sum = np.zeros((n_heads, seq_length), dtype = np.int32)
        else:
            shift_sum = np.zeros((n_heads, seq_length), dtype = np.float32)

        # Calculate the number of shifts required to updated the already accumulated sum
        # Make sure to do use round-half-up instead of round-half-to-even
        if integerize:
            max_shift = np.floor((current_max - global_max) * eps_max + 0.5 + np.finfo(np.float32).eps)
        else:
            max_shift = (current_max - global_max) * eps_max

        # Update all shift values where new maximum is larger
        shift_sum[current_max > global_max] = max_shift[current_max > global_max]

        # Updated all maximums where they changed
        global_max[current_max > global_max] = current_max[current_max > global_max]

        # Find the difference between the maximum and x in the current part of the row
        if integerize:
            diff = np.repeat(global_max, width).reshape(
                n_heads, seq_length, width) - x[..., 0 + i * width:width + i * width].astype(np.int32)
        else:
            diff = np.repeat(global_max, width).reshape(
                n_heads, seq_length, width) - x[..., 0 + i * width:width + i * width].astype(np.float32)

        # Shift the values by B-log2B -> multiply by B/2**B = log2e*eps_x
        # Make sure to do use round-half-up instead of round-half-to-even
        if integerize:
            shift = np.floor(diff * eps_max + 0.5 + np.finfo(np.float32).eps).astype(np.int32)
        else:
            shift = diff * eps_max

        # Calculate exponential sum over the current part of the row and scale it by 2**10 to prevent underflow
        if integerize:
            # exp_sum = np.sum(2**8 >> shift, -1) # or
            exp_sum = np.floor(np.sum(2**8 / 2**shift, axis = -1))
        else:
            exp_sum = np.sum(1 / 2**shift, axis = -1)

        # Update the accumulated sum and add the accumulation over the current part of the row
        if integerize:
            exp_partial_sum = np.floor((exp_partial_sum / 2**shift_sum)) + exp_sum
        else:
            exp_partial_sum = (exp_partial_sum / 2**(shift_sum.astype(np.float32))) + exp_sum

    ## STAGE 2: Calculate the softmax activation
    # Invert the partial sum
    # WIESEP: Scale Softmax to 127
    # The Softmax values are maximum 127 as sumdotp modules can only do signed-signed operations for now. This is a temporary fix until sumdotp is fixed.
    if integerize:
        exp_partial_sum_inverse = np.floor((2**7 - 1) * 2**8 / exp_partial_sum).astype(np.int32)
    else:
        exp_partial_sum_inverse = 1 / exp_partial_sum

    # Find the difference between the maximum and x
    diff = np.repeat(global_max, seq_length).reshape(n_heads, seq_length, seq_length) - x.astype(np.int32)

    # Shift the values by B-log2B -> multiply by B/2**B = log2e*eps_x
    # Make sure to do use round-half-up instead of round-half-to-even
    if integerize:
        shift = np.floor(diff * eps_max + 0.5 + np.finfo(np.float32).eps).astype(np.int32)
    else:
        shift = diff * eps_max

    # Calculate the activation value
    if integerize:
        # A_partial_softmax[0] = np.repeat(exp_partial_sum_inverse, seq_length).reshape(seq_length, seq_length) >> shift
        return np.floor(
            np.repeat(exp_partial_sum_inverse, seq_length).reshape(n_heads, seq_length, seq_length) / 2**shift).astype(
                np.int8)
    else:
        return np.repeat(exp_partial_sum_inverse, seq_length).reshape(n_heads, seq_length, seq_length) / 2**shift


def realSoftmax(A_requant, integerize = True):
    n_heads = A_requant.shape[-3]

    B = 8
    log2e = np.log2(np.exp(1))
    eps_x = B / (2**B * log2e)

    if integerize:
        x = A_requant * eps_x
    else:
        x = A_requant.astype(np.float64)

    exp = np.exp(x - np.max(x, axis = 2).reshape(n_heads, -1, 1))
    if integerize:
        return (exp / exp.sum(axis = 2).reshape(n_heads, -1, 1) * (2**7 - 1)).astype(A_requant.dtype)
    else:
        return exp / exp.sum(axis = 2).reshape(n_heads, -1, 1)


def softermaxStep1(A_requant, integerize = True):
    # Get the number of heads
    n_heads = A_requant.shape[-3]

    # Define the number of bits
    B = 8

    # Calculate the logarithm of the base e
    log2e = np.log2(np.exp(1))

    # Calculate the scaling factor
    eps_x = B / (2**B * log2e)

    # Scale the input values if integerize is True
    if integerize:
        x = A_requant * eps_x
    else:
        x = A_requant.astype(np.float64)
    
    # Calculate the exponential values
    exp = 2**(x - np.max(x, axis = 2).reshape(n_heads, -1, 1))

    # Normalize the exponential values to get the softmax probabilities
    if integerize:
        # Scale the probabilities to the range [0, 2^7 - 1]
        return (exp / exp.sum(axis = 2).reshape(n_heads, -1, 1) * (2**7 - 1)).astype(A_requant.dtype)
    else:
        return exp / exp.sum(axis = 2).reshape(n_heads, -1, 1)
    
def softermaxStep2(A_requant, integerize = True):
    # Get the number of heads
    n_heads = A_requant.shape[-3]

    # Define the number of bits
    B = 8

    # Calculate the logarithm of the base e
    log2e = np.log2(np.exp(1))

    # Calculate the scaling factor
    eps_x = B / (2**B * log2e)

    # Scale the input values if integerize is True
    if integerize:
        x = A_requant * eps_x
    else:
        x = A_requant.astype(np.float64)
    
    max_values = -np.inf * np.ones((n_heads, x.shape[1], x.shape[2]))
    exp_result = np.zeros((n_heads, x.shape[1], x.shape[2]))
    sum = 0
    
    # Calculate the exponential values
    # Loop over each head
    for i in range(n_heads):
        # Loop over each row
        for j in range(x.shape[1]):
            # Loop over each element in the row
            max_val = x[i, j, 0]
            for k in range(x.shape[2]):
                # Calculate the exponentiation
                if x[i, j, k] > max_val:
                    max_val = x[i, j, k]
                max_values[i, j, k] = max_val
                if(k==0):
                    exp_result[i, j, k] = 1
                    sum = exp_result[i, j, k]
                else:
                    exp_result[i, j, k] = 2 ** (x[i, j, k] - max_values[i, j, k])
                    sum = exp_result[i, j, k] + sum * 2 ** (max_values[i, j, k-1] - max_values[i, j, k])

    #ÇNormalize the exponential values to get the softmax probabilities
    # Loop over each head
    for i in range(n_heads):
        # Loop over each row
        for j in range(x.shape[1]):
            # Loop over each element in the row
            for k in range(x.shape[2]):
                exp_result[i, j, k] = (exp_result[i, j, k]* 2**(max_values[i, j, k] - max_values[i, j, -1])) / sum


    # Normalize the exponential values to get the softmax probabilities
    if integerize:
        # Scale the probabilities to the range [0, 2^7 - 1]
        return  exp_result * (2**7 - 1).astype(A_requant.dtype)
    else:
        return exp_result
    
def softermaxStep3(A_requant, integerize = True):
    # Get the number of heads
    n_heads = A_requant.shape[-3]

    # Define the number of bits
    B = 8

    # Calculate the logarithm of the base e
    log2e = np.log2(np.exp(1))

    # Calculate the scaling factor
    eps_x = B / (2**B * log2e)

    # Scale the input values if integerize is True
    if integerize:
        x = A_requant * eps_x
    else:
        x = A_requant.astype(np.float64)
    
    max_values = -np.inf * np.ones((n_heads, x.shape[1], x.shape[2]))
    exp_result = np.zeros((n_heads, x.shape[1], x.shape[2]))
    sum = 0
    
    # Calculate the exponential values
    # Loop over each head
    for i in range(n_heads):
        # Loop over each row
        for j in range(x.shape[1]):
            # Loop over each element in the row
            max_val = x[i, j, 0]
            for k in range(x.shape[2]):
                # Calculate the exponentiation
                if x[i, j, k] > max_val:
                    max_val = x[i, j, k]
                max_values[i, j, k] = int(max_val)
                if(k==0):
                    exp_result[i, j, k] = 1
                    sum = exp_result[i, j, k]
                else:
                    exp_result[i, j, k] = 2 ** (x[i, j, k] - max_values[i, j, k])
                    sum = int(exp_result[i, j, k] + sum) >> int(max_values[i, j, k] - max_values[i, j, k-1])

    #Normalize the exponential values to get the softmax probabilities
    # Loop over each head
    for i in range(n_heads):
        # Loop over each row
        for j in range(x.shape[1]):
            # Loop over each element in the row
            for k in range(x.shape[2]):
                exp_result[i, j, k] = (int(exp_result[i, j, k]) >> int(max_values[i, j, -1] - max_values[i, j, k]) )/ sum


    # Normalize the exponential values to get the softmax probabilities
    if integerize:
        # Scale the probabilities to the range [0, 2^7 - 1]
        return  exp_result * (2**7 - 1)
    else:
        return exp_result
