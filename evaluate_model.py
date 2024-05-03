import operator
import torch
#import torch._dynamo
import torch.nn as nn
from datasets import load_dataset
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.fx import symbolic_trace


from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, MobileBertConfig
from datasets import load_dataset
import numpy as np
from transformers import DataCollatorWithPadding
import evaluate
import argparse
import copy
import traceback
from functools import partial
from typing import Callable, List, Literal
from passes import ApproximateSoftmaxPass
from optimum.fx.optimization import Transformation


############### QuantLib Imports ###############################################
import quantlib.backends.deeploy as deeploy
import quantlib.editing.lightweight as qlw
import quantlib.editing.lightweight.rules as qlr
import quantlib.editing.fx as qlfx
import quantlib.algorithms as qla
from quantlib.editing.fx.passes.pact import PACTInclusiveTracer, PACT_symbolic_trace, PACT_OPS, PACT_OPS_INCLUSIVE
from quantlib.editing.fx.passes.general import ModularizeActivationsPass
from quantlib.editing.fx.util.tracing import LeafTracer, custom_symbolic_trace
from quantlib.editing.lightweight.rules.filters import NameFilter
from quantlib.algorithms.pact.pact_ops import (PACTITAMax,
                                               PACTSoftmax,
                                               PACTIntegerSoftmax,
                                               PACTUnsignedAct,
                                               PACTITAPartialMax)

N_LEVELS_ACTS = 2**8
UPPER_PERCENTILE = 99.9
LOWER_PERCENTILE = 0.1
EPOCHS = 4

schedule = {1: "start", (EPOCHS - 1): ["freeze"]}
actSchedule = {1: "start", (EPOCHS - 1): ["freeze"]}
epsSchedule = {(EPOCHS - 2): 'start'}


softmax_cfg = {
    "mode": "I-BERT",
    "n_levels": 2**12,
    "init_clip": "max",
    "leaky": 0.0,
    "learn_clip": True,
    "lower_percentile": LOWER_PERCENTILE,
    "num_bins": 2**12,
    "rounding": True,
    "tqt": True,
    "upper_percentile": UPPER_PERCENTILE,
    "act_kind": "identity"
}



def eval_model(model = MobileBertForSequenceClassification.from_pretrained("Alireza1044/mobilebert_sst2")):
    sst2_dataset = load_dataset("sst2")
    evaluate_dataset = sst2_dataset["validation"]

    tokenizer = MobileBertTokenizer.from_pretrained("Alireza1044/mobilebert_sst2")
    # print(MobileBertConfig())
    correct_predictions = 0
    total_examples = len(evaluate_dataset)

    for example in evaluate_dataset:
        inputs = tokenizer(example["sentence"], return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs)["logits"]

        predicted_class_id = logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]
        true_label = example["label"]

        if predicted_class_id == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_examples
    print("Accuracy:", accuracy)




def quantize_softmax(model_name="Alireza1044/mobilebert_sst2",
                     dataset_name="sst2",
                     n_train=128,
                     n_test=128,
                     batch_size=4,
                     epochs=1,
                     verbose=0):

    # Load dataset
    sst2 = load_dataset("stanfordnlp/sst2")

    print(sst2["train"][0])

    tokenizer = MobileBertTokenizer.from_pretrained(
        "Alireza1044/mobilebert_sst2")

    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_sst2 = sst2.map(preprocess_function, batched=True)
    print(tokenized_sst2['train'][0])
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MobileBertForSequenceClassification.from_pretrained(
        "Alireza1044/mobilebert_sst2",
        num_labels=2,
        id2label=id2label,
        label2id=label2id).to(device)

    # Trace model

    print("[=== Step 1 : Trace Model ===")
    # torch._dynamo.reset()

    # TODO: understand tracing
    traced = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
    )

    nodes_list = qlw.LightweightGraph.build_nodes_list(
        traced, leaf_types=PACT_OPS_INCLUSIVE)

    print("=== Original Model ===")
    lwg = qlw.LightweightGraph(traced)
    lwe = qlw.LightweightEditor(lwg)
    for lwn in nodes_list:
        print("    {:30s} {}".format(lwn.name, lwn.type_))
    print()

    print(type(model), type(traced))
    _passes = []
    # _passes.append()
    _passes.append(ApproximateSoftmaxPass(**softmax_cfg))
    traced = ModularizeActivationsPass().apply(traced)
    print("=== Modularize Activations ===")
    nodes_list = qlw.LightweightGraph.build_nodes_list(
        traced, leaf_types=PACT_OPS_INCLUSIVE)
    for lwn in nodes_list:
        print("    {:30s} {}".format(lwn.name, lwn.type_))
    print()

    traced = ApproximateSoftmaxPass(**softmax_cfg).apply(traced)

    print("=== Quantized Model ===")
    nodes_list = qlw.LightweightGraph.build_nodes_list(
        traced, leaf_types=PACT_OPS_INCLUSIVE)
    for lwn in nodes_list:
        print("    {:30s} {}".format(lwn.name, lwn.type_))
    print()

    # print(traced.print_readable())
    train_activations(n_train, n_test, batch_size, sst2, device, traced)
    # print(traced)

    return traced


def train_activations(n_train, n_test, batch_size, sst2, device, traced):

    sst2['train'] = sst2['train'].select(range(n_train))
    act_list = [
        i for i in traced.modules() if isinstance(i, qla.pact._PACTActivation)
    ]
    print(act_list)

    verbose = 1
    if verbose > 0:
        _verbose = True
    else:
        _verbose = False

    actController = qla.pact.PACTActController(act_list,
                                               actSchedule,
                                               verbose=_verbose)

    optimizer = torch.optim.Adam(traced.parameters(), lr=0)
    loss_fn = nn.CrossEntropyLoss()
    traced.train()
    tokenizer = MobileBertTokenizer.from_pretrained(
        "Alireza1044/mobilebert_sst2")
    # Define the collate function for DataLoader
    #TODO: fix bugs
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def _collate_fn(batch, device):
        sentences = [
            tokenizer(item["sentence"],
                      truncation=True,
                      padding='longest',
                      return_tensors='pt') for item in batch
        ]
        labels = torch.tensor([item['label'] for item in batch])
        return sentences, labels

    # Create DataLoader
    dataloader_train_batch = DataLoader(sst2["train"],
                                        batch_size=batch_size,
                                        collate_fn=partial(_collate_fn,
                                                           device=device))
    dataloader_test_batch = DataLoader(sst2["validation"],
                                       batch_size=batch_size,
                                       collate_fn=partial(_collate_fn,
                                                          device=device))
    num_test_examples = len(sst2['validation'])
    num_train_examples = len(sst2["train"])

    # def _getAbsMinAbsMax(tensor, n_levels=N_LEVELS_ACTS):
    #     _max = tensor.max()
    #     _min = tensor.min()

    #     if _max == 0 and _min == 0:
    #         _max = 1

    #     absMax = max(_max, torch.abs(_min))

    #     if min == 0:
    #         absMin = 0
    #     else:
    #         absMin = -absMax / ((n_levels // 2) - 1) * (n_levels // 2)

    #     return absMin, absMax

    # if n_test > 0 and n_test < num_test_examples:
    #     num_test_examples = n_test

    # if n_train > 0 and n_train < num_train_examples:
    #     num_train_examples = n_train

    num_train_batches = int(np.ceil(num_train_examples / batch_size))

    torch.autograd.set_detect_anomaly(True)

    for epoch in range(EPOCHS):
        actController.step_pre_training_epoch(epoch, optimizer)

        correct = 0
        total = 0
        eps_in = 1
        with tqdm(total=num_train_examples, desc="Train",
                  leave=False) as pbar_batch:
            for input_batch, label_batch in dataloader_train_batch:
                input_batch = input_batch
                label_batch = label_batch.to(device)

                # absMin, absMax = _getAbsMinAbsMax(input_batch)
                # input_batch_rounded = torch.trunc(
                #     torch.clamp(input_batch, min=absMin, max=absMax) /
                #     eps_in) * eps_in
                actController.step_pre_training_batch(epoch, optimizer)
                outputs_list = [None] * len(input_batch)
                predicted_list = [None] * len(input_batch)
                for i in range(len(input_batch)):
                    outputs_list[i] = traced(**input_batch[i])["logits"]
                    _, predicted_list[i] = torch.max(outputs_list[i], 1)

                total += label_batch.size(0)
                correct += (
                    torch.tensor(predicted_list) == label_batch).sum().item()
                accuracy = correct / total

                loss = loss_fn(torch.cat(outputs_list, dim=0), label_batch)

                # optimizer.zero_grad()  # clear gradients
                # loss.backward()  # gradient computation
                # optimizer.step()  # gradient descent

                pbar_batch.set_description(
                    f'Train [{epoch+1:02d}/{EPOCHS:02d}] -- Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Batch'
                )
                pbar_batch.update(label_batch.size(0))

                if total >= num_train_examples:
                    break
                pbar_batch.close()
                print(
                    f' Train [{epoch+1:02d}/{EPOCHS:02d}] -- Accuracy: {accuracy:.4f}'
                )


model = quantize_softmax()

print("Evaluating quantized model...")
eval_model(model)
print("Evaluating unquantized model...")
eval_model()
