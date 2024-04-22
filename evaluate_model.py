from datasets import load_dataset
from transformers import MobileBertTokenizer, MobileBertForSequenceClassification
from evaluate import load
import numpy as np
from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModelForSequenceClassification
import torch




# Load dataset
sst2_dataset = load_dataset("sst2")
#sst2_dataset = sst2_dataset.remove_columns("idx")

sst2_dataset = sst2_dataset.map(lambda examples: {"sentence": examples["sentence"], "label": examples["label"]}, batched=True)
print(sst2_dataset)
# Preprocessing function
def preprocess_function(examples):
  return {
      "sentence": [sentence.lower() for sentence in examples["sentence"]],
      "label": examples["label"],
  }

# Tokenization function
tokenizer = MobileBertTokenizer.from_pretrained("Alireza1044/mobilebert_sst2")

# Tokenize data
def tokenize_function(examples):
  encoding = tokenizer(examples["sentence"], padding=True, truncation=True,return_tensors="pt")
  # Access and return the necessary elements
  return {
      'input_ids': encoding['input_ids'],
      'attention_mask': encoding['attention_mask'],
      'label': torch.tensor(examples["label"])
  }



# Encode data (batched)
encoded_train = sst2_dataset["train"].map(tokenize_function, batched=True)
encoded_val = sst2_dataset["validation"].map(tokenize_function, batched=True)
encoded_test = sst2_dataset["test"].map(tokenize_function, batched=True)

# Load model
model = MobileBertForSequenceClassification.from_pretrained("Alireza1044/mobilebert_sst2")

# Define metric
metric = load("accuracy")

def compute_metrics(pred):
  logits = pred.logits
  return metric.compute(predictions=logits, references="Positive")

# Evaluate model (iterating through batches)
for batch in encoded_train:
  # Access tokenized elements from the batch dictionary
  input_ids = batch['input_ids']
  attention_mask = batch['attention_mask']

  # Call the model with unpacked arguments
  print(input_ids)
  print(attention_mask)
  print(torch.FloatTensor(sst2_dataset["train"]["label"]))
  train_preds = model(**batch)
  print(train_preds)
  train_results = compute_metrics(train_preds)
  break  # Early stopping

# Similar loops for validation and test sets
for batch in encoded_val:
  input_ids = batch['input_ids']
  attention_mask = batch['attention_mask']

  val_preds = model(input_ids=torch.LongTensor(input_ids).unsqueeze(0), attention_mask=torch.FloatTensor(attention_mask).unsqueeze(0))
  val_results = compute_metrics(val_preds)
  break

for batch in encoded_test:
  input_ids = batch['input_ids']
  attention_mask = batch['attention_mask']

  test_preds = model(input_ids=torch.LongTensor(input_ids).unsqueeze(0), attention_mask=torch.FloatTensor(attention_mask).unsqueeze(0))
  test_results = compute_metrics(test_preds)
  break

print("Train Accuracy:", train_results["accuracy"])
print("Validation Accuracy:", val_results["accuracy"])
print("Test Accuracy:", test_results["accuracy"])

