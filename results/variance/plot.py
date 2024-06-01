import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data for I-BERT and ITA-Partial on 6 bits for sst2
data = [
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9105504587155964},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9105504587155964},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9071100917431193},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9013761467889908},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9094036697247706},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9002293577981652},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9013761467889908},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.8979357798165137},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.908256880733945},
    {'Bits': 6, 'Mode': 'I-BERT', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
    {'Bits': 6, 'Mode': 'ITA-Partial', 'True_Accuracy': 0.9025229357798165},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Calculate mean and standard deviation for each mode
df_grouped = df.groupby(['Bits', 'Mode'])['True_Accuracy'].agg(['mean', 'std']).reset_index()

# Plot the results
plt.figure(figsize=(10, 6))

# Error bars for standard deviation
for mode in df_grouped['Mode'].unique():
    df_mode = df_grouped[df_grouped['Mode'] == mode]
    plt.errorbar(df_mode['Bits'], df_mode['mean'], yerr=df_mode['std'], fmt='o', capsize=5, label=f'{mode} Mean Accuracy with Std Dev')

# Add scatter points for all individual results
for mode in df['Mode'].unique():
    df_mode = df[df['Mode'] == mode]
    plt.scatter([6] * len(df_mode), df_mode['True_Accuracy'], alpha=0.5, label=f'{mode} Individual Accuracies')

# Add the baseline
baseline_value = 0.9036697247706422
plt.axhline(y=baseline_value, color='r', linestyle='--', label='Baseline Accuracy')

plt.xlabel('Bits')
plt.ylabel('Accuracy')
plt.title('Variance Analysis of True Accuracy for I-BERT and ITA-Partial on sst2 (6 Bits)')
plt.legend()
plt.grid(True)

# Save the figure
plt.tight_layout()
plt.savefig('results/variance/plot.png')
print("Plot has been saved to variance_analysis_ibert_ita_partial_sst2.png")
