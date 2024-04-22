import torch
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from datasets import load_dataset

sst2_dataset = load_dataset("sst2")
evaluate_dataset = sst2_dataset["train"]

tokenizer = MobileBertTokenizer.from_pretrained("./transformers_model")
model = MobileBertForSequenceClassification.from_pretrained("./transformers_model")

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



