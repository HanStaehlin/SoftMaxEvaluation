# MobileBERT Quantization Framework

This repository contains an implementation of integer-only quantization for MobileBERT models, with a focus on efficient inference on resource-constrained devices. The framework supports various quantization modes including I-BERT and ITA (Integer-only Transformer Architecture).

## Features

- Support for multiple GLUE benchmark tasks
- Configurable bit-width quantization (default: 6-bit)
- Multiple quantization modes (I-BERT, ITA, ITA-Partial)
- Histogram visualization for quantization analysis
- Comprehensive evaluation metrics for each GLUE task
- Softmax approximation and integer-only operations

## Requirements

```
torch
transformers
datasets
scipy
sklearn
tqdm
matplotlib
quantlib (custom library for quantization)
```

## Supported Models and Datasets

The framework supports various MobileBERT models fine-tuned on GLUE tasks:

- SST-2 (Sentiment Analysis)
- CoLA (Linguistic Acceptability)
- MNLI (Natural Language Inference)
- MRPC (Paraphrase Detection)
- QNLI (Question Answering NLI)
- QQP (Question Pair Similarity)
- RTE (Recognizing Textual Entailment)
- STS-B (Semantic Similarity)
- WNLI (Winograd NLI)

## Usage

### Basic Usage

```bash
python main.py --config mobilebert_sst2 --bits 3
```

### Configuration Options

- `--config`: Model-dataset configuration (e.g., mobilebert_sst2, mobilebert_cola)
- `--bits`: Number of bits for quantization (default: 3)

### Global Settings

Key parameters can be configured at the top of the script:

```python
Plot_Histograms = False  # Enable/disable histogram plotting
N_LEVELS_ACTS = 2**6    # Quantization levels
UPPER_PERCENTILE = 99.9 # Clipping threshold
LOWER_PERCENTILE = 10   # Lower bound threshold
EPOCHS = 5             # Training epochs
BATCH_SIZE = 1         # Batch size
N_TRAIN = 10          # Number of training samples
```

## Quantization Process

1. **Model Tracing**: The framework first traces the model using PyTorch's FX
2. **Activation Modularization**: Separates activation functions for quantization
3. **Softmax Approximation**: Implements integer-only softmax
4. **Training**: Fine-tunes the quantized model
5. **Integerization**: Converts to fully integer operations

## Evaluation Metrics

- Accuracy (for classification tasks)
- F1 Score (for MRPC, QQP)
- Matthews Correlation Coefficient (for CoLA)
- Pearson Correlation (for STS-B)

## Results

Results are saved to `results/reduced_bitwidth/ITAV4.txt` and include:
- Dataset name
- Model configuration
- Quantization mode
- Bit width
- Clipping bounds
- Metrics for original, fake quantized, and true quantized models

## Visualization

The framework supports two types of histogram visualizations:
- Pre-quantization distributions
- Post-quantization integer value distributions

Enable visualization by setting:
```python
Plot_Histograms = True
Plot_Histograms_Integer = True
```

## Contributing

Please feel free to submit issues and pull requests for:
- Additional model support
- New quantization schemes
- Performance improvements
- Bug fixes


## Acknowledgments

This project utilizes components from:
- HuggingFace Transformers
- PyTorch
- GLUE Benchmark
