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

from fx import HFLeafTracer, SimpleInterpreter

# Hyperparameters
N_LEVELS_ACTS = 2**8
UPPER_PERCENTILE = 99.9
LOWER_PERCENTILE = 0.1
EPOCHS = 4
num_layers = 4
schedule = {1: "start", (EPOCHS - 1): ["freeze"]}
actSchedule = {1: "start", (EPOCHS - 1): ["freeze"]}
epsSchedule = {(EPOCHS - 2): 'start'}


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
    "act_kind": "identity"
}
consmax_cfg = {
    "mode": "consmax",
    "n_levels": N_LEVELS_ACTS,
    "init_clip": "max",
    "leaky": 0.0,
    "learn_clip": True,
    "lower_percentile": LOWER_PERCENTILE,
    "num_bins": 2**12,
    "rounding": True,
    "tqt": True,
}



def eval_model(model=MobileBertForSequenceClassification.from_pretrained(
    "Alireza1044/mobilebert_sst2")):
    sst2_dataset = load_dataset("sst2")
    evaluate_dataset = sst2_dataset["validation"]

    tokenizer = MobileBertTokenizer.from_pretrained(
        "Alireza1044/mobilebert_sst2")
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
    return accuracy




def quantize_softmax(model_name="Alireza1044/mobilebert_sst2",
                     dataset_name="sst2",
                     n_train=4,
                     n_test=128,
                     batch_size=128,
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

    htraced = symbolic_trace(
        model,
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        tracer_cls=HFLeafTracer,
    )
    htraced = ModularizeActivationsPass().apply(htraced)
    htraced = ApproximateSoftmaxPass(**softmax_cfg).apply(htraced)

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

    embedder = MobileBertEmbeddings(modelConf)

    # Instanticate the dataloader
    dataloader_train = DataLoader(dataset_train,
                                  batch_size=batch_size,
                                  collate_fn=partial(_collate_fn,
                                                     device=device))
    dataloader_test = DataLoader(dataset_test,
                                 batch_size=batch_size,
                                 collate_fn=partial(_collate_fn,
                                                    device=device))

    # Get sample batch
    train_batch = next(iter(dataloader_train))
    test_batch = next(iter(dataloader_test))

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
    nodes_list = qlw.LightweightGraph.build_nodes_list(
        traced, leaf_types=PACT_OPS_INCLUSIVE)

    print("=== Original Model ===")
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

    # traced = symbolic_trace(
    #     traced,
    #     input_names=["input_ids", "attention_mask", "token_type_ids"],
    #     tracer_cls=HFLeafTracer,
    # )

    print("=== Quantized Model ===")

    # torch._dynamo.reset()
    # graphs: List[torch.fx.GraphModule] = []

    # for param in model.parameters():
    #     param.requires_grad = False

    # try:
    #     model_fn = torch.compile(backend=partial(dynamo_graph_extract_compiler,
    #                                              model),
    #                              dynamic=False)(model)
    #     _ = model_fn(fakeBatch[0], fakeBatch[1], train_batch_head_mask)
    # except Exception as e:
    #     print("[MobileBERT] === PyTorch Network (non-tracable) ===\n", model)
    #     print("[MobileBERT] === Error ===\n", e)
    #     if verbose > 0:
    #         traceback.print_exc()
    #     import IPython
    #     IPython.embed()
    #     exit(-1)

    # gm = graphs[0]
    # gm.graph.eliminate_dead_code()
    # gm.recompile()
    # htraced = HFLeafTracer(htraced)
    nodes_list = qlw.LightweightGraph.build_nodes_list(
        traced, leaf_types=PACT_OPS_INCLUSIVE)
    traced.graph.print_tabular()

    train_activations(n_train, n_test, batch_size, sst2, device, htraced)
    # print(traced)

    return htraced


def train_activations(n_train, n_test, batch_size, sst2, device, traced):

    sst2['train'] = sst2['train'].select(range(n_train))
    act_list = [
        i for i in traced.modules() if isinstance(i, qla.pact._PACTActivation)
    ]
    print(act_list)
    _verbose = 1
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

                optimizer.zero_grad()  # clear gradients
                loss.backward()  # gradient computation
                optimizer.step()  # gradient descent

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
#     axs[i].plot([1, 4, 8, 16, 32], result, label=f"{2**(i+2)} bits")
#     axs[i].set_title(f"Performance with {2**(i+2)} bits")
#     axs[i].set_xlabel("Number of Training Examples")
#     axs[i].set_ylabel("Evaluation Result")
#     axs[i].legend()

# # Save the visualization to a file
# plt.tight_layout()
# plt.savefig("results_visualization.png")
# plt.show()

model = quantize_softmax()
print("Evaluating quantized model...")

eval_model(model)
print("Evaluating unquantized model...")
eval_model()
