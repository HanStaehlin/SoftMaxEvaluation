import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification, MobileBertConfig
from datasets import load_dataset
import numpy as np
from transformers import DataCollatorWithPadding
import evaluate

############### QuantLib Imports ###############################################
import quantlib.backends.deeploy as deeploy
import quantlib.editing.lightweight as qlw
import quantlib.editing.lightweight.rules as qlr
import quantlib.editing.fx as qlfx
import quantlib.algorithms as qla

from quantlib.editing.fx.util.tracing import LeafTracer



sst2_dataset = load_dataset("sst2")
evaluate_dataset = sst2_dataset["validation"]

tokenizer = MobileBertTokenizer.from_pretrained("Alireza1044/mobilebert_sst2")
model = MobileBertForSequenceClassification.from_pretrained("Alireza1044/mobilebert_sst2")
print(MobileBertConfig())
correct_predictions = 0
total_examples = len(evaluate_dataset)

for example in evaluate_dataset:
    inputs = tokenizer(example["sentence"], return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    predicted_label = model.config.id2label[predicted_class_id]
    true_label = example["label"]

    if predicted_class_id == true_label:
        correct_predictions += 1

accuracy = correct_predictions / total_examples
print("Accuracy:", accuracy)


def quantize_model(model_name, dataset_name, n_train = 128, n_test = 128, batch_size = 4, epochs = 1, verbose = 0):
    
    # Load dataset
    sst2 = load_dataset("sst2")

    print(sst2["train"][0])

    tokenizer = MobileBertTokenizer.from_pretrained("Alireza1044/mobilebert_sst2")

    def preprocess_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_imdb = sst2.map(preprocess_function, batched=True)

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
        "google/mobilebert-uncased", num_labels=2, id2label=id2label, label2id=label2id
    ).to(device)

    # Trace model

    print("[=== Step 1 : Trace Model ===")
    torch._dynamo.reset()


    graphs: List[torch.fx.GraphModule] = []

    def dynamo_graph_extract_compiler(model, gm: GraphModule, inputs: List[torch.Tensor]) -> Callable:
        gm = foldConstant(gm, matchSizeNode, inputs)
        gm = foldConstant(gm, matchShapeNode, inputs)
        gm = foldConstant(gm, matchGetAttrNode, inputs[0])
        gm = delistifyInputs(gm)

        graphs.append(gm)
        return gm.forward

    # Compile model and extract FX graph
    try:
        model_fn = torch.compile(backend = partial(dynamo_graph_extract_compiler, model), dynamic = False)(model)
        _ = model_fn(data_train_batch[0])
    except Exception as e:
        print("[DINOv2] === PyTorch Network (non-tracable) ===\n", model)
        print("[DINOv2] === Error ===\n", e)
        if verbose > 0:
            traceback.print_exc()
        exit(-1)

    gm = graphs[0]

    if verbose > 1:
        print(gm)
        print(gm.graph.print_tabular())

    # Export ONNX model
    torch.onnx.export(gm, args = data_train_batch[0], f = "network.onnx")
