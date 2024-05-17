import argparse
import ipdb
import os
from functools import partial
from typing import Callable, List

import torch
import torch.nn as nn
from datasets import load_dataset
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, precision_score, recall_score)
from torch.fx import GraphModule
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (MobileBertForSequenceClassification, MobileBertTokenizer, MobileBertModel)
import matplotlib.pyplot as plt

from typing import Literal
from transformers.utils import logging

import numpy as np

############### QuantLib Imports ###############################################
import quantlib.algorithms as qla
from fx import HFLeafTracer
from passes import ApproximateSoftmaxPass, IntegerizeSoftmaxPass, CustomAnnotateEpsPass
from quantlib.editing.fx.passes.general import (ModularizeActivationsPass, RetracePass)
from quantlib.editing.fx.passes.pact import PACT_OPS, PACT_symbolic_trace
from quantlib.editing.fx.passes.pact.integerize import (AnnotateEpsPass, IntegerizeBNActPass)
from quantlib.editing.fx.util.tracing import LeafTracer
from utils import _getAdhocEpsList
from fx import SimpleInterpreter, HistogramInterpreter

# Hyperparameters
N_LEVELS_ACTS = 2**8
UPPER_PERCENTILE = 99.9
LOWER_PERCENTILE = 0.1
EPOCHS = 5
BATCH_SIZE = 1
NumHiddenLayers = 1
schedule = {1: "start", (EPOCHS - 1): ["freeze"]}
actSchedule = {1: "start", (EPOCHS - 1): ["freeze"]}
epsSchedule = {(EPOCHS - 2): 'start'}

fixed_max_length = 150  # Set a fixed max length for all inputs


def pearson_correlation(y_true, y_pred):
    correlation, _ = pearsonr(y_true, y_pred)
    return correlation


metrics = {
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "mcc": matthews_corrcoef,  # Matthews Correlation Coefficient
    "pearson": pearson_correlation,
}

loss_functions = {
    'cola': CrossEntropyLoss(),  # CoLA might be better with a binary cross-entropy loss
    'sst2': CrossEntropyLoss(),  # SST-2 is a binary classification (positive/negative)
    'mrpc': CrossEntropyLoss(),  # MRPC is also a binary classification task
    'stsb': MSELoss(),  # STS-B requires a regression loss
    'mnli': CrossEntropyLoss(),  # MNLI involves multi-class classification
    'qnli': CrossEntropyLoss(),  # QNLI is a binary classification task
    'qqp': CrossEntropyLoss(),  # QQP can be approached with binary classification as well
    'rte': CrossEntropyLoss(),  # RTE is binary classification
    'wnli': CrossEntropyLoss(),  # WNLI is also binary classification
    'ax': CrossEntropyLoss(),  # AX (diagnostic task) depending on setup might need cross-entropy
}

model_dataset_configs = {
    "mobilebert_sst2": {
        "model_name": "Alireza1044/mobilebert_sst2",
        "dataset_name": "sst2",
        "tokenizer": "Alireza1044/mobilebert_sst2",
        "metrics": "accuracy",  # Since SST2 is about sentiment classification
    },
    "mobilebert_cola": {
        "model_name": "Alireza1044/mobilebert_cola",
        "dataset_name": "cola",
        "tokenizer": "Alireza1044/mobilebert_cola",
        "metrics": "mcc",  # MCC is recommended for CoLA
    },
    "mobilebert_mnli": {
        "model_name": "Alireza1044/mobilebert_mnli",
        "dataset_name": "mnli",
        "tokenizer": "Alireza1044/mobilebert_mnli",
        "metrics": "accuracy",  # MNLI involves textual entailment
    },
    "mobilebert_mrpc": {
        "model_name": "Alireza1044/mobilebert_mrpc",
        "dataset_name": "mrpc",
        "tokenizer": "Alireza1044/mobilebert_mrpc",
        "metrics": "f1",  # MRPC often uses F1 score
    },
    "mobilebert_qnli": {
        "model_name": "Alireza1044/mobilebert_qnli",
        "dataset_name": "qnli",
        "tokenizer": "Alireza1044/mobilebert_qnli",
        "metrics": "accuracy",  # QNLI is a binary classification task
    },
    "mobilebert_qqp": {
        "model_name": "Alireza1044/mobilebert_qqp",
        "dataset_name": "qqp",
        "tokenizer": "Alireza1044/mobilebert_qqp",
        "metrics": "f1",  # QQP also often uses F1 score for evaluation
    },
    "mobilebert_rte": {
        "model_name": "Alireza1044/mobilebert_rte",
        "dataset_name": "rte",
        "tokenizer": "Alireza1044/mobilebert_rte",
        "metrics": "accuracy",  # RTE is a smaller entailment dataset
    },
    "mobilebert_stsb": {
        "model_name": "Alireza1044/mobilebert_stsb",
        "dataset_name": "stsb",
        "tokenizer": "Alireza1044/mobilebert_stsb",
        "metrics": "pearson",  # STS-B uses Pearson correlation
    },
    "mobilebert_wnli": {  # No Model available
        "model_name": "Alireza1044/mobilebert_wnli",
        "dataset_name": "wnli",
        "tokenizer": "Alireza1044/mobilebert_wnli",
        "metrics": "accuracy",  # WNLI is based on correct pronoun resolution
    },
    "mobilebert_ax": {  # Not a standard dataset, but a diagnostic set for MNLI
        "model_name": "Alireza1044/mobilebert_multinli",
        "dataset_name": "ax",
        "tokenizer": "Alireza1044/mobilebert_multinli",
        "metrics": "accuracy",  # AX is an analysis set for MNLI
    }
}

softmax_cfg = {
    "mode": "I-BERT",
    "n_levels": N_LEVELS_ACTS,
    "init_clip": "percentile",  #try out
    "leaky": 0.0,
    "learn_clip": True,
    "lower_percentile": LOWER_PERCENTILE,
    "num_bins": 2**12,
    "rounding": True,
    "tqt": True,
    "upper_percentile": UPPER_PERCENTILE,
    "act_kind": "identity",
}


def _print_tabular(gm: GraphModule):
    """
        Prints the intermediate representation of the graph in tabular
        format with quanitzation metadata.
        """
    try:
        from tabulate import tabulate
    except ImportError:
        print("`print_tabular` relies on the library `tabulate`, "
              "which could not be found on this machine. Run `pip "
              "install tabulate` to install the library.")
        raise

    def quant_info(node, prop: Literal['eps', 'n_levels', 'signed'] = 'eps'):
        if 'quant' in node.meta:
            _repr = ''
            if prop == 'eps':
                _repr += str(node.meta['quant'].eps_in)
                _repr += ' -> '
                _repr += str(node.meta['quant'].eps_out)
            elif prop == 'n_levels':
                _repr += str(np.ceil(np.log2(node.meta['quant'].n_levels_in)).astype(int))
                _repr += ' -> '
                _repr += str(np.ceil(np.log2(node.meta['quant'].n_levels_out)).astype(int))
            elif prop == 'signed':
                _repr += str(node.meta['quant'].signed_in)
                _repr += ' -> '
                _repr += str(node.meta['quant'].signed_out)
            return _repr
        else:
            return '{}'

    def class_info(node):
        if node.op == 'call_module':
            return gm.get_submodule(node.target).__class__.__name__
        else:
            return ''

    node_specs = [[
        n.op,
        class_info(n), n.name, n.target, n.args, n.kwargs,
        quant_info(n, 'n_levels'),
        quant_info(n, 'signed')
    ] for n in gm.graph.nodes]
    print(tabulate(node_specs, headers = ['opcode', 'class', 'name', 'target', 'args', 'kwargs', 'n_levels', 'signed']))


def eval_model(config, model = None, batch_size = BATCH_SIZE, n_test = 512):
    if model is None:
        model = MobileBertForSequenceClassification.from_pretrained(config['model_name'])
    model.eval()

    # Load the appropriate dataset
    dataset_name = config['dataset_name']
    tokenizer = MobileBertTokenizer.from_pretrained(config['tokenizer'])
    dataset = load_dataset("nyu-mll/glue", dataset_name)

    # Select the appropriate validation set

    if dataset_name == "mnli":
        evaluate_dataset = dataset["validation_matched"]
    else:
        evaluate_dataset = dataset["validation"]

    if n_test != -1:
        evaluate_dataset = evaluate_dataset.select(range(n_test))

    predictions, labels = [], []
    for example in tqdm(evaluate_dataset, desc = "Evaluating"):
        # Prepare inputs based on the dataset type
        if dataset_name in ["cola", "sst2"]:
            inputs = tokenizer(example["sentence"],
                               return_tensors = "pt",
                               padding = 'max_length',
                               truncation = True,
                               max_length = fixed_max_length)
        elif dataset_name in ["mrpc", "stsb", "rte", "wnli"]:
            inputs = tokenizer(example["sentence1"],
                               example["sentence2"],
                               return_tensors = "pt",
                               padding = 'max_length',
                               truncation = True,
                               max_length = fixed_max_length)
        elif dataset_name in ["qqp"]:
            inputs = tokenizer(example["question1"],
                               example["question2"],
                               return_tensors = "pt",
                               padding = 'max_length',
                               truncation = True,
                               max_length = fixed_max_length)
        elif dataset_name in ["qnli"]:
            inputs = tokenizer(example["question"],
                               example["sentence"],
                               return_tensors = "pt",
                               padding = 'max_length',
                               truncation = True,
                               max_length = fixed_max_length)
        elif dataset_name in ["mnli", "mnli_matched", "mnli_mismatched", "ax"]:
            inputs = tokenizer(example["premise"],
                               example["hypothesis"],
                               return_tensors = "pt",
                               padding = 'max_length',
                               truncation = True,
                               max_length = fixed_max_length)

        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            predicted_class_id = logits.argmax(dim = -1).item()

        predictions.append(predicted_class_id)
        labels.append(example["label"])

    # Calculate the selected metric
    selected_metric_key = config['metrics']
    selected_metric = metrics[selected_metric_key]
    result = selected_metric(labels, predictions)
    print(f"{selected_metric_key.capitalize()}:", result)
    return result


def _collate_fn(batch, tokenizer, task_type):
    if task_type in ['sst2', 'cola']:  # Tasks with a single sentence input
        batch_inputs = tokenizer([item["sentence"] for item in batch],
                                 return_tensors = "pt",
                                 padding = 'max_length',
                                 truncation = True,
                                 max_length = fixed_max_length)
    elif task_type in ['mrpc', 'stsb', 'rte', 'wnli']:  # Tasks with two sentences
        batch_inputs = tokenizer([item["sentence1"] for item in batch], [item["sentence2"] for item in batch],
                                 return_tensors = "pt",
                                 padding = 'max_length',
                                 truncation = True,
                                 max_length = fixed_max_length)
    elif task_type in ['qqp']:  # Tasks with two questions
        batch_inputs = tokenizer([item["question1"] for item in batch], [item["question2"] for item in batch],
                                 return_tensors = "pt",
                                 padding = 'max_length',
                                 truncation = True,
                                 max_length = fixed_max_length)
    elif task_type in ['mnli', 'mnli_matched', 'mnli_mismatched', 'ax']:  # Tasks with premise and hypothesis
        batch_inputs = tokenizer([item["premise"] for item in batch], [item["hypothesis"] for item in batch],
                                 return_tensors = "pt",
                                 padding = 'max_length',
                                 truncation = True,
                                 max_length = fixed_max_length)
    elif task_type == 'qnli':  # QNLI with question and context sentence
        batch_inputs = tokenizer([item["question"] for item in batch], [item["sentence"] for item in batch],
                                 return_tensors = "pt",
                                 padding = 'max_length',
                                 truncation = True,
                                 max_length = fixed_max_length)

    labels = torch.tensor([item['label'] for item in batch])
    return {**batch_inputs, 'labels': labels}


def quantize_softmax(config, dataloader, n_train = 10, n_test = 128, epochs = 1, verbose = 0):
    model = MobileBertForSequenceClassification.from_pretrained(config['model_name'])
    dataset_name = config['dataset_name']
    dataset = load_dataset("nyu-mll/glue", dataset_name)

    # modelConf = model.config
    # modelConf.num_hidden_layers = NumHiddenLayers
    # model = MobileBertModel(modelConf)
    device = torch.device('cpu')

    # Trace model
    model.eval()

    # Get sample batch
    train_batch = next(iter(dataloader))
    print("[MobileBERT] ======= Step 1 : Trace Model =======")
    torch._dynamo.reset()
    graphs: List[torch.fx.GraphModule] = []

    def dynamo_graph_extract_compiler(model, gm: GraphModule, inputs: List[torch.Tensor]) -> Callable:
        graphs.append(gm)
        return gm.forward

    for param in model.parameters():
        param.requires_grad = False

    model_fn = torch.compile(backend = partial(dynamo_graph_extract_compiler, model), dynamic = False)(model)
    _ = model_fn(train_batch["input_ids"], train_batch["attention_mask"], train_batch["token_type_ids"])

    gm = graphs[0]
    gm.graph.eliminate_dead_code()
    gm.recompile()

    traced = gm

    print("=== Original Model ===")
    # print(traced)
    # traced.graph.print_tabular()
    #eval_model(config, traced, n_test = n_test)

    print("=== Modularize Activations ===")
    traced_mod = ModularizeActivationsPass().apply(traced)
    traced_approx = ApproximateSoftmaxPass(**softmax_cfg).apply(traced_mod)
    traced_approx = PACT_symbolic_trace(traced_approx)

    # print(traced_approx)
    # print(_print_tabular(traced_approx))
    #eval_model(config, traced_approx, n_test = n_test)

    train_activations(config, n_train, n_test, dataset, device, traced_approx, dataloader)

    return traced_approx


def get_loss_function(task_type):
    return loss_functions.get(task_type, CrossEntropyLoss())  # Default to CrossEntropyLoss if not specified


def train_activations(config, n_train, n_test, dataset, device, traced, dataloader_train_batch):
    task_type = config['dataset_name']
    dataset['train'] = dataset['train'].select(range(n_train))

    SOFTMAX_EVAL_Tracer = LeafTracer(leaf_types = list(PACT_OPS))

    act_list = [i for i in traced.modules() if isinstance(i, qla.pact._PACTActivation)]
    eps_list = [i for i in traced.modules() if isinstance(i, qla.pact._PACTEps)]

    actController = qla.pact.PACTActController(modules = act_list,
                                               schedule = actSchedule,
                                               init_clip_hi = 6.,
                                               init_clip_lo = -6.,
                                               verbose = True)

    annotateEpsPass = CustomAnnotateEpsPass()
    epsController = qla.pact.PACTEpsController(fx_model = traced,
                                               modules = eps_list,
                                               schedule = epsSchedule,
                                               tracer = SOFTMAX_EVAL_Tracer,
                                               eps_pass = annotateEpsPass,
                                               verbose = True)

    loss_fn = get_loss_function(task_type)
    traced.train()

    num_train_examples = len(dataset["train"])
    torch.autograd.set_detect_anomaly(True)

    sp = SimpleInterpreter(traced)

    for epoch in range(EPOCHS):
        actController.step_pre_training_epoch(epoch)
        epsController.step_pre_training_epoch(epoch)
        traced.train()
        correct = 0
        total = 0

        hi = HistogramInterpreter(traced)
        batch=next(iter(dataloader_train_batch))
        #hi.propagate(epoch, batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
        with tqdm(total = num_train_examples, desc = "Train", leave = False) as pbar_batch:
            for batch in dataloader_train_batch:
                # Ensure all input tensors are moved to the correct device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = inputs.pop('labels')

                # WIESEP: For debugging purposes, we can use the SimpleInterpreter to check the intermediate activations and output of the model
                outputs = traced(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
                #outputs = sp.propagate(inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])

                actController.step_pre_training_batch()
                epsController.step_pre_training_batch()

                # Adjust this line based on the output structure
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs

                loss = loss_fn(logits, labels)

                predicted = logits.argmax(dim = 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                accuracy = correct / total

                pbar_batch.set_description(
                    f'Train [{epoch+1}/{EPOCHS}] -- Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
                pbar_batch.update(labels.size(0))

                if total >= num_train_examples:
                    break
            pbar_batch.close()
            print(f'Train [{epoch+1}/{EPOCHS}] -- Accuracy: {accuracy:.4f}')
        # ipdb.set_trace()


def IntergerizePass(model):
    model = RetracePass(PACT_symbolic_trace).apply(model)
    model = CustomAnnotateEpsPass(verbose = True).apply(model)
    model = IntegerizeBNActPass().apply(model)
    model = IntegerizeSoftmaxPass().apply(model)
    return model


if __name__ == "__main__":
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser(description = 'Test different MobileBERT models and datasets.')
    parser.add_argument('--config',
                        type = str,
                        help = 'Configuration key for model and dataset',
                        required = False,
                        default = 'mobilebert_sst2')
    args = parser.parse_args()

    config = model_dataset_configs[args.config]

    print(f"Quantizing model for {config['model_name']} on {config['dataset_name']} dataset...")

    dataset = load_dataset("nyu-mll/glue", config['dataset_name'])
    tokenizer = MobileBertTokenizer.from_pretrained(config['tokenizer'])

    dataloader = DataLoader(dataset["train"],
                            batch_size = BATCH_SIZE,
                            collate_fn = lambda x: _collate_fn(x, tokenizer, config['dataset_name']))

    model = MobileBertForSequenceClassification.from_pretrained(config['model_name'])
    model_fq = quantize_softmax(config, dataloader)
    model_tq = IntergerizePass(model_fq)

    print("=== Evaluating Original Model ===")
    eval_model(config, model, n_test = -1)

    print("=== Evaluating Fake-Quant Model ===")
    eval_model(config, model_fq, n_test = -1)

    print("=== Evaluating True-Quant Model ===")
    eval_model(config, model_tq, n_test = -1)

    sp = SimpleInterpreter(model_tq)
    batch=next(iter(dataloader))
    sp.propagate(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
    from pprint import pprint
    pprint([{k: [v.min(), v.max(),v]} for k, v in sp.env.items() if "softmax" in k])

    plt.figure(figsize=(10, 6))
    for k, v in sp.env.items():
        if "softmax" in k:
            plt.hist(v[v>-128].flatten().numpy(), bins=30, color='blue', alpha=0.7)  
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Tensor Values')
    # Save the plot to a file
    plt.savefig('tensor_histogram.png')
    plt.close()  # Close the plot explicitly after savingq
    plt.show()  # Display the plot