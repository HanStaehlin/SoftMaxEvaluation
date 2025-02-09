from datasets import load_dataset
from transformers import pipeline


sst2_dataset = load_dataset("sst2")
evaluate_dataset = sst2_dataset["validation"]
# Todo: For each example in the dataset, feed the sentence to the model and get the prediction, then compare the prediction to the label and keep track of the accuracy.

classifier = pipeline("sentiment-analysis", model="Alireza1044/mobilebert_sst2")

accuracy = 0
for example in evaluate_dataset:
    input_text = example["sentence"]
    label = example["label"]
    
    prediction = classifier(input_text)
    predicted_label = prediction[0]["label"]
    
    if predicted_label == 'positive':
        if label == 1:
            accuracy += 1
    elif predicted_label == 'negative':
        if label == 0:
            accuracy += 1

total_examples = len(evaluate_dataset)
accuracy_percentage = (accuracy / total_examples) * 100

print(f"Accuracy: {accuracy_percentage:.2f}%")
