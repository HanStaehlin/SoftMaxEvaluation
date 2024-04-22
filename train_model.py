import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

sst2 = load_dataset("sst2")

print(sst2["train"][0])



tokenizer = AutoTokenizer.from_pretrained("google/mobilebert-uncased")

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

model = AutoModelForSequenceClassification.from_pretrained(
    "google/mobilebert-uncased", num_labels=2, id2label=id2label, label2id=label2id
).to(device)

training_args = TrainingArguments(
    output_dir="./transformers_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
