# ----------------------------------------------------------------------
#
# File: main.py
#
# Last edited: 01.02.2024
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

############### Other Libraries ################################################
import argparse
import traceback
from functools import partial
from typing import Callable, List

############### PyTorch Imports ################################################
import torch
import torch.nn as nn
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from tqdm import tqdm
############### HuggingFace Imports ############################################

############### QuantLib Imports ###############################################
import quantlib.backends.deeploy as deeploy
import quantlib.editing.lightweight as qlw
import quantlib.editing.lightweight.rules as qlr
############### Constants ######################################################
SEED = 42

############### Private Declarations ##############################################
model_name = "google/mobilebert-uncased"
dataset_name = "stanfordnlp/sst2"


# Define the collate function for DataLoader
def _collate_fn(batch, device):
    images = torch.stack([item['input_pixels'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch]).to(device)
    return images, labels


############### Public Functions ###############################################
def evaluate_model(model_name, dataset_name, num_samples = 100, batch_size = 32, verbose = 0):
    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model and tokenizer
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

    # Load the dataset
    dataset = load_dataset(dataset_name, streaming = True, trust_remote_code = True, split = 'validation')

    # Randomly shuffle the dataset
    dataset = dataset.shuffle(seed = SEED, buffer_size = num_samples)


    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size = batch_size, collate_fn = partial(_collate_fn, device = device))

    # Evaluation loop
    model.eval()
    correct = 0
    total = 0
    sample_count = 0
    with torch.no_grad(), tqdm(total = num_samples) as pbar:
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            sample_count += labels.size(0)
            if sample_count >= num_samples:
                break
            accuracy = correct / total
            pbar.set_description(f'Accuracy: {accuracy:.4f}')
            pbar.update(labels.size(0))

    accuracy = correct / total
    print(f'Final Accuracy: {accuracy:.4f}')


def quantize_model(model_name, dataset_name, num_samples = 100, epochs = 1, verbose = 0):
    # Check if GPU is available
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    # Load the model and tokenizer
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    processor = AutoImageProcessor.from_pretrained(model_name)

    # Load the dataset
    dataset = load_dataset(dataset_name, streaming = True, trust_remote_code = True, split = 'validation')

    # Randomly shuffle the dataset
    # dataset = dataset.shuffle(seed = SEED, buffer_size = num_samples)

    # Preprocess the dataset
    dataset = dataset.map(partial(_preprocess_images, processor = processor, device = device))

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size = 1, collate_fn = partial(_collate_fn, device = device))

    # Get sample image
    image = next(iter(dataloader))[0]

    model.eval()

    # Trace model
    try:
        torch._dynamo.reset()

        # Allow PACT ops in graph
        allow_ops_in_graph(DINOV2_OPS)

        model_fn = torch.compile(backend = partial(dinov2_quant_backend, model), dynamic = False)(model)
        model_quant = model_fn(image)
    except Exception as e:
        print("[QuantLab] === PyTorch Network (non-tracable) ===\n", model)
        print("[QuantLab] === Error ===\n", e)
        if verbose > 0:
            traceback.print_exc()
        exit(-1)

    print()


############### Main Functions #################################################

if __name__ == '__main__':
    evaluate_model(model_name, dataset_name, num_samples = 100, verbose = 0)

