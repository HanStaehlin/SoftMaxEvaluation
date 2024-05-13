import operator
import torch
# import torch._dynamo
import torch.nn as nn
from torch.fx import GraphModule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.utils.fx import symbolic_trace, get_concrete_args
import matplotlib.pyplot as plt


from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, MobileBertConfig
from transformers.models.mobilebert.modeling_mobilebert import MobileBertEncoder, MobileBertEmbeddings, MobileBertModel
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
from utils import print_tabular, _getAdhocEpsList, roundTensors, delistify, OutputProjectionITA, getMAEPercent
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr
############### QuantLib Imports ###############################################
import quantlib.backends.deeploy as deeploy
import quantlib.editing.lightweight as qlw
import quantlib.editing.lightweight.rules as qlr
import quantlib.editing.fx as qlfx
import quantlib.algorithms as qla
from quantlib.editing.fx.passes.pact import PACTInclusiveTracer, PACT_symbolic_trace, PACT_OPS, PACT_OPS_INCLUSIVE
from quantlib.editing.fx.passes.general import ModularizeActivationsPass
from quantlib.editing.fx.passes.pact.integerize import IntegerizeSoftmaxPass, IntegerizePACTNetPass
from quantlib.editing.fx.util.tracing import LeafTracer, custom_symbolic_trace
from quantlib.editing.lightweight.rules.filters import NameFilter
from quantlib.algorithms.pact.pact_ops import (PACTITAMax,
                                               PACTSoftmax,
                                               PACTIntegerSoftmax,
                                               PACTUnsignedAct,
                                               PACTITAPartialMax)

from fx import HFLeafTracer, SimpleInterpreter
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss

# Hyperparameters
N_LEVELS_ACTS = 2**8
UPPER_PERCENTILE = 99.9
LOWER_PERCENTILE = 0.1
EPOCHS = 4
num_layers = 1
schedule = {1: "start", (EPOCHS - 1): ["freeze"]}
actSchedule = {1: "start", (EPOCHS - 1): ["freeze"]}
epsSchedule = {(EPOCHS - 2): 'start'}


def pearson_correlation(y_true, y_pred):
    # Pearson correlation expects two arrays of the same length
    # `pearsonr` returns a tuple (correlation, p-value), so we extract just the correlation part
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
    'cola': CrossEntropyLoss(),   # CoLA might be better with a binary cross-entropy loss
    'sst2': CrossEntropyLoss(),   # SST-2 is a binary classification (positive/negative)
    'mrpc': CrossEntropyLoss(),   # MRPC is also a binary classification task
    'stsb': MSELoss(),            # STS-B requires a regression loss
    'mnli': CrossEntropyLoss(),   # MNLI involves multi-class classification
    'qnli': CrossEntropyLoss(),   # QNLI is a binary classification task
    'qqp': CrossEntropyLoss(),    # QQP can be approached with binary classification as well 
    'rte': CrossEntropyLoss(),    # RTE is binary classification
    'wnli': CrossEntropyLoss(),   # WNLI is also binary classification
    'ax': CrossEntropyLoss(),     # AX (diagnostic task) depending on setup might need cross-entropy
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
    "mobilebert_wnli": { # No Model available
        "model_name": "Alireza1044/mobilebert_wnli",
        "dataset_name": "wnli",
        "tokenizer": "Alireza1044/mobilebert_wnli",
        "metrics": "accuracy",  # WNLI is based on correct pronoun resolution
    },
    "mobilebert_ax": { # Not a standard dataset, but a diagnostic set for MNLI
        "model_name": "Alireza1044/mobilebert_multinli",
        "dataset_name": "ax",
        "tokenizer": "Alireza1044/mobilebert_multinli",
        "metrics": "accuracy",  # AX is an analysis set for MNLI
    }
}

softmax_cfg = {
    "mode": "I-BERT",
    "n_levels": N_LEVELS_ACTS,
    "init_clip": "max",
    "leaky": 0.0,
    "learn_clip": True,
    "lower_percentile": LOWER_PERCENTILE,
    "num_bins": 2**12,
    "rounding": True,
    "tqt": True,
    "upper_percentile": UPPER_PERCENTILE,
    "act_kind": "identity",

}







def eval_model(config, model=None):
    if model is None:
        model = MobileBertForSequenceClassification.from_pretrained(config['model_name'])
    model.eval()

    # Load the appropriate dataset
    dataset_name = config['dataset_name']
    tokenizer = MobileBertTokenizer.from_pretrained(config['tokenizer'])
    dataset = load_dataset("nyu-mll/glue", dataset_name)

    # Select the appropriate validation set
    if dataset_name == "mnli":
        evaluate_dataset = dataset["validation_matched"].select(range(200))
    else:
        evaluate_dataset = dataset["validation"].select(range(200))

    predictions, labels = [], []
    for example in tqdm(evaluate_dataset, desc="Evaluating"):
        # Prepare inputs based on the dataset type
        if dataset_name in ["cola", "sst2"]:
            inputs = tokenizer(example["sentence"], return_tensors="pt", padding=True, truncation=True)
        elif dataset_name in ["mrpc", "stsb", "rte", "wnli"]:
            inputs = tokenizer(example["sentence1"], example["sentence2"], return_tensors="pt", padding=True, truncation=True)
        elif dataset_name in ["qqp"]:
            inputs = tokenizer(example["question1"], example["question2"], return_tensors="pt", padding=True, truncation=True)
        elif dataset_name in ["qnli"]:
            inputs = tokenizer(example["question"], example["sentence"], return_tensors="pt", padding=True, truncation=True)
        elif dataset_name in ["mnli", "mnli_matched", "mnli_mismatched", "ax"]:
            inputs = tokenizer(example["premise"], example["hypothesis"], return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"]
            predicted_class_id = logits.argmax(dim=-1).item()

        predictions.append(predicted_class_id)
        labels.append(example["label"])

    # Calculate the selected metric
    selected_metric_key = config['metrics']
    selected_metric = metrics[selected_metric_key]
    result = selected_metric(labels, predictions)
    print(f"{selected_metric_key.capitalize()}:", result)
    return result




def quantize_softmax(config,
                     n_train=1,
                     n_test=128,
                     batch_size=1,
                     epochs=1,
                     verbose=0):
    model = MobileBertForSequenceClassification.from_pretrained(config['model_name'])
    dataset_name = config['dataset_name']
    tokenizer_name = config['tokenizer']

    dataset = load_dataset("nyu-mll/glue",dataset_name)
    tokenizer = MobileBertTokenizer.from_pretrained(tokenizer_name)
    # Load dataset

    print(dataset["train"][0])

    # def compute_metrics(eval_pred):
    #     predictions, labels = eval_pred
    #     predictions = np.argmax(predictions, axis=1)
    #     return accuracy.compute(predictions=predictions, references=labels)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MobileBertForSequenceClassification.from_pretrained(config['model_name']).to(device)

    # Trace model

    print("[=== Step 1 : Trace Model ===")
    # torch._dynamo.reset()

    htraced = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        tracer_cls=HFLeafTracer,
    )
    htraced = ModularizeActivationsPass().apply(htraced)
    htraced = ApproximateSoftmaxPass(**softmax_cfg).apply(htraced)
    #htraced = IntegerizeSoftmaxPass(**softmax_cfg).apply(htraced)


    stateDict = model.state_dict()
    modelConf = model.config
    modelConf.num_hidden_layers = num_layers

    def _collate_fn(batch, device):
        texts = [item['text'] for item in batch]
        input_ids = torch.stack(
            [torch.Tensor(item['input_ids']) for item in batch]).to(device)
        attention_masks = torch.stack([
            torch.Tensor(item['attention_mask']) for item in batch
        ]).to(device)
        labels = torch.tensor(
            [torch.Tensor([item['label']]) for item in batch]).to(device)
        return texts, input_ids, attention_masks, labels

    dataset_train = load_dataset("emo",
                                 streaming=False,
                                 trust_remote_code=True,
                                 split='train',
                                 keep_in_memory=True)
    dataset_test = load_dataset("emo",
                                streaming=False,
                                trust_remote_code=True,
                                split='test',
                                keep_in_memory=True)

    # Tokenize, truncate and pad the dataset
    dataset_train = dataset_train.map(
        lambda x: tokenizer(x['text'],
                            truncation=True,
                            max_length=modelConf.max_length,
                            padding='max_length'),
        batched=True,
        keep_in_memory=True)
    # dataset_train = dataset_train.shuffle(seed = 4232423452)
    dataset_test = dataset_test.map(
        lambda x: tokenizer(x['text'],
                            truncation=True,
                            max_length=modelConf.max_length,
                            padding='max_length'),
        batched=True,
        keep_in_memory=True)
    # dataset_test = dataset_test.shuffle(seed = 4232423452)
    mobileBertModel = MobileBertModel(modelConf)
    model = MobileBertEncoder(modelConf)  # JUNGVI: First Quantize the encoder
    model.load_state_dict(stateDict, strict=False)

    model.eval()
    for MBertLayer in model.layer:
        MBertLayer.attention.output.LayerNorm = torch.nn.Identity()
        MBertLayer.output.LayerNorm = torch.nn.Identity()
        MBertLayer.output.bottleneck.LayerNorm = torch.nn.Identity()
        MBertLayer.bottleneck.input.LayerNorm = torch.nn.Identity()
        MBertLayer.bottleneck.attention.LayerNorm = torch.nn.Identity()
        for ffnLayer in MBertLayer.ffn:
            ffnLayer.output.LayerNorm = torch.nn.Identity()

    embedder = MobileBertEmbeddings(modelConf)

    # Instanticate the dataloader
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  collate_fn=partial(_collate_fn,
                                                     device=device))
    # dataloader_test = DataLoader(dataset_test,
    #                              batch_size=batch_size,
    #                              collate_fn=partial(_collate_fn,
    #                                                 device=device))

    # Get sample batch
    train_batch = next(iter(dataloader_train))
    #test_batch = next(iter(dataloader_test))

    train_batch_embed = embedder(
        input_ids=train_batch[1].type(torch.LongTensor))
    train_batch_head_mask = mobileBertModel.get_head_mask(
        None, modelConf.num_hidden_layers)
    train_batch_attention_mask_extended = mobileBertModel.get_extended_attention_mask(
        train_batch[2].type(torch.LongTensor), train_batch[1].size())
    # train_output = model(hidden_states=train_batch_embed, attention_mask=attention_mask_extended, head_mask=head_mask)

    fakeBatch = [train_batch_embed, train_batch_attention_mask_extended]
    eps_in = tuple(_getAdhocEpsList(256, *fakeBatch))
    eps_in[1].data = torch.tensor(
        1.)  # JUNGVI: replace eps of mask with 1.0 as it is inf otherwise
    fakeBatch = roundTensors(fakeBatch, eps_in)
    fakeBatch[1][fakeBatch[1] != 0] = -256 // 2

    # goldenOutput = model(fakeBatch[0], fakeBatch[1], train_batch_head_mask)
    print("[MobileBERT] ======= Step 1 : Trace Model =======")

    torch._dynamo.reset()
    graphs: List[torch.fx.GraphModule] = []

    def dynamo_graph_extract_compiler(model, gm: GraphModule,
                                      inputs: List[torch.Tensor]) -> Callable:
        # foldConstant(gm, matchGetAttrNode, *inputs)
        graphs.append(gm)
        return gm.forward

    for param in model.parameters():
        param.requires_grad = False

    try:
        model_fn = torch.compile(backend=partial(dynamo_graph_extract_compiler,
                                                 model),
                                 dynamic=False)(model)
        _ = model_fn(fakeBatch[0], fakeBatch[1], train_batch_head_mask)
    except Exception as e:
        print("[MobileBERT] === PyTorch Network (non-tracable) ===\n", model)
        print("[MobileBERT] === Error ===\n", e)
        if verbose > 0:
            traceback.print_exc()
        import IPython
        IPython.embed()
        exit(-1)

    gm = graphs[0]
    gm.graph.eliminate_dead_code()
    gm.recompile()

    traced = gm

    print("=== Original Model ===")
    traced.graph.print_tabular()

    print(type(model), type(traced))

    print("=== Modularize Activations ===")
    traced = ModularizeActivationsPass().apply(traced)
    traced.graph.print_tabular()

    print("=== Exchange Softmax ===")
    traced = ApproximateSoftmaxPass(**softmax_cfg).apply(traced)
    traced_approx = copy.deepcopy(traced)
    SOFTMAX_EVAL_PACT_OPS = PACT_OPS
    SOFTMAX_EVAL_Tracer = LeafTracer(leaf_types=list(SOFTMAX_EVAL_PACT_OPS))
    SOFTMAX_EVAL_PACT_symbolic_trace = partial(custom_symbolic_trace,
                                               tracer=SOFTMAX_EVAL_Tracer)
    traced_approx = SOFTMAX_EVAL_PACT_symbolic_trace(traced_approx)
    traced_approx.graph.print_tabular()

    train_activations(config, n_train, n_test, batch_size, dataset, device, htraced)
    # print(traced)

    return htraced

def _collate_fn(batch, tokenizer, task_type):
    if task_type in ['sst2', 'cola']:  # Tasks with a single sentence input
        batch_inputs = tokenizer([item["sentence"] for item in batch],
                                 return_tensors="pt", padding=True, truncation=True)
    elif task_type in ['mrpc', 'stsb', 'rte', 'wnli']:  # Tasks with two sentences
        batch_inputs = tokenizer([item["sentence1"] for item in batch],
                                 [item["sentence2"] for item in batch],
                                 return_tensors="pt", padding=True, truncation=True)
    elif task_type in ['qqp']:  # Tasks with two questions
        batch_inputs = tokenizer([item["question1"] for item in batch],
                                 [item["question2"] for item in batch],
                                 return_tensors="pt", padding=True, truncation=True)
    elif task_type in ['mnli', 'mnli_matched', 'mnli_mismatched', 'ax']:  # Tasks with premise and hypothesis
        batch_inputs = tokenizer([item["premise"] for item in batch],
                                 [item["hypothesis"] for item in batch],
                                 return_tensors="pt", padding=True, truncation=True)
    elif task_type == 'qnli':  # QNLI with question and context sentence
        batch_inputs = tokenizer([item["question"] for item in batch],
                                 [item["sentence"] for item in batch],
                                 return_tensors="pt", padding=True, truncation=True)

    labels = torch.tensor([item['label'] for item in batch])

    # Return a dictionary with the same keys as expected by the model
    return {**batch_inputs, 'labels': labels}


def get_loss_function(task_type):
    return loss_functions.get(task_type, CrossEntropyLoss())  # Default to CrossEntropyLoss if not specified

def train_activations(config, n_train, n_test, batch_size, dataset, device, traced):
    task_type = config['dataset_name']
    tokenizer = MobileBertTokenizer.from_pretrained(config['tokenizer'])
    dataset['train'] = dataset['train'].select(range(n_train))
    act_list = [i for i in traced.modules() if isinstance(i, qla.pact._PACTActivation)]
    _verbose = 1
    actController = qla.pact.PACTActController(act_list, actSchedule, verbose=_verbose)

    optimizer = torch.optim.Adam(traced.parameters(), lr=0)
    loss_fn = get_loss_function(task_type)
    traced.train()

    dataloader_train_batch = DataLoader(dataset["train"], batch_size=batch_size, collate_fn=lambda x: _collate_fn(x, tokenizer, task_type))

    num_train_examples = len(dataset["train"])
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(EPOCHS):
        actController.step_pre_training_epoch(epoch, optimizer)
        traced.train()
        correct = 0
        total = 0
        with tqdm(total=num_train_examples, desc="Train", leave=False) as pbar_batch:
            for batch in dataloader_train_batch:
                # Ensure all input tensors are moved to the correct device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                labels = inputs.pop('labels')  # Remove labels from inputs and use separately as targets

                #optimizer.zero_grad()
                outputs = traced(**inputs)["logits"]  # Pass the rest of the inputs to the model
                loss = loss_fn(outputs, labels)
                #loss.backward()
                #optimizer.step()

                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                accuracy = correct / total

                pbar_batch.set_description(f'Train [{epoch+1}/{EPOCHS}] -- Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
                pbar_batch.update(labels.size(0))

                if total >= num_train_examples:
                    break
            pbar_batch.close()
            print(f'Train [{epoch+1}/{EPOCHS}] -- Accuracy: {accuracy:.4f}')


# results = []
# for bit in [2, 4, 8, 16]:
#     c = []
#     softmax_cfg["n_levels"] = 2**bit
#     for n_train in [1, 4, 8, 16, 32]:
#         print(f"Training with {n_train} examples and {bit} bits")
#         model = quantize_softmax(n_train=n_train)
#         c.append(eval_model(model))
#     results.append(c)
# print(results)

# # Visualize the results
# fig, axs = plt.subplots(len(results), sharex=True, figsize=(10, 8))
# for i, result in enumerate(results):
#     axs[i].plot([1, 4, 8, 16, 32], result, label=f"{2**(i)} bits")
#     axs[i].set_title(f"Performance with {2**(i)} bits")
#     axs[i].set_xlabel("Number of Training Examples")
#     axs[i].set_ylabel("Evaluation Result")
#     axs[i].legend()

#Save the visualization to a file
# plt.tight_layout()
# plt.savefig("results_visualization_without_tqt.png")
# plt.show()

# model = quantize_softmax()
# print("Evaluating quantized model...")
# print(softmax_cfg["mode"])
# eval_model(model)
# print("Evaluating unquantized model...")
# eval_model()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test different MobileBERT models and datasets.')
    parser.add_argument('--config', type=str, help='Configuration key for model and dataset', required=False, default='mobilebert_qqp')
    args = parser.parse_args()

    config = model_dataset_configs[args.config]

    print(f"Quantizing model for {config['model_name']} on {config['dataset_name']} dataset...")
    model = quantize_softmax(config)
    print(f"Evaluating quantized model on {config['dataset_name']}...")
    eval_model(config,model)
    print("Evaluating unquantized model...")
    eval_model(config)
