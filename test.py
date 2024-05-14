import torch
from transformers import MobileBertTokenizer, MobileBertForPreTraining
from torch.fx import GraphModule
from typing import List, Callable
from functools import partial

# Load tokenizer and model
tokenizer = MobileBertTokenizer.from_pretrained('google/mobilebert-uncased')
model = MobileBertForPreTraining.from_pretrained('google/mobilebert-uncased')

# Prepare input text
input_text = "Example text to process"
inputs = tokenizer(input_text, return_tensors="pt")

# Set up torch.dynamo
torch._dynamo.reset()
graphs: List[torch.fx.GraphModule] = []

def dynamo_graph_extract_compiler(model, gm: GraphModule, inputs: List[torch.Tensor]) -> Callable:
    graphs.append(gm)
    return gm.forward

# Set model to eval mode and disable gradients
model.eval()
for param in model.parameters():
    param.requires_grad = False

# Compile model with dynamo
model_fn = torch.compile(backend=partial(dynamo_graph_extract_compiler, model), dynamic=False)(model)

# Run the model to extract the graph
_ = model_fn(inputs)

# Now 'graphs' contains the traced graph
print(graphs[0])  # You can inspect the first graph or any specific one
