import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data for the plots
data = [
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'Accuracy', 'Value': 0.9036697247706422},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'Accuracy', 'Value': 0.9036697247706422},
    {'Dataset': 'cola', 'Model': 'Alireza1044/mobilebert_cola', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'Mcc', 'Value': 0.51728018358102},
    {'Dataset': 'cola', 'Model': 'Alireza1044/mobilebert_cola', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'Mcc', 'Value': 0.5110339437874821},
    {'Dataset': 'rte', 'Model': 'Alireza1044/mobilebert_rte', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'Accuracy', 'Value': 0.5703971119133574},
    {'Dataset': 'rte', 'Model': 'Alireza1044/mobilebert_rte', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'Accuracy', 'Value': 0.631768953068592},
    {'Dataset': 'mrpc', 'Model': 'Alireza1044/mobilebert_mrpc', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'F1', 'Value': 0.7412008281573499},
    {'Dataset': 'mrpc', 'Model': 'Alireza1044/mobilebert_mrpc', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Largest_Negative_Int', 'Metric': 'F1', 'Value': 0.8752079866888519},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Nan', 'Metric': 'Accuracy', 'Value': 0.9025229357798165},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Nan', 'Metric': 'Accuracy', 'Value': 0.9071100917431193},
    {'Dataset': 'cola', 'Model': 'Alireza1044/mobilebert_cola', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Nan', 'Metric': 'Mcc', 'Value': 0.51728018358102},
    {'Dataset': 'cola', 'Model': 'Alireza1044/mobilebert_cola', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Nan', 'Metric': 'Mcc', 'Value': 0.5202123212283704},
    {'Dataset': 'rte', 'Model': 'Alireza1044/mobilebert_rte', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Nan', 'Metric': 'Accuracy', 'Value': 0.5956678700361011},
    {'Dataset': 'rte', 'Model': 'Alireza1044/mobilebert_rte', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Nan', 'Metric': 'Accuracy', 'Value': 0.628158844765343},
    {'Dataset': 'mrpc', 'Model': 'Alireza1044/mobilebert_mrpc', 'Mode': 'I-BERT', 'Bits': 6.0, 'Approach': 'Nan', 'Metric': 'F1', 'Value': 0.7427385892116183},
    {'Dataset': 'mrpc', 'Model': 'Alireza1044/mobilebert_mrpc', 'Mode': 'I-BERT', 'Bits': 8.0, 'Approach': 'Nan', 'Metric': 'F1', 'Value': 0.8760330578512396},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # Increased width to 24 inches

# Plot for sst2
ax = axs[0, 0]
df_filtered = df[(df['Dataset'] == 'sst2') & (df['Metric'] == 'Accuracy')]
for approach in df_filtered['Approach'].unique():
    df_approach = df_filtered[df_filtered['Approach'] == approach]
    ax.scatter(df_approach['Bits'], df_approach['Value'], label=approach)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits')
ax.set_ylabel('Accuracy')
ax.set_title('sst2 - Accuracy vs Bits')
ax.legend()
ax.grid(True)

# Plot for cola
ax = axs[0, 1]
df_filtered = df[(df['Dataset'] == 'cola') & (df['Metric'] == 'Mcc')]
for approach in df_filtered['Approach'].unique():
    df_approach = df_filtered[df_filtered['Approach'] == approach]
    ax.scatter(df_approach['Bits'], df_approach['Value'], label=approach, s=100)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits')
ax.set_ylabel('Mcc')
ax.set_title('cola - Mcc vs Bits')
ax.legend()
ax.grid(True)

# Plot for rte
ax = axs[1, 0]
df_filtered = df[(df['Dataset'] == 'rte') & (df['Metric'] == 'Accuracy')]
for approach in df_filtered['Approach'].unique():
    df_approach = df_filtered[df_filtered['Approach'] == approach]
    ax.scatter(df_approach['Bits'], df_approach['Value'], label=approach)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits')
ax.set_ylabel('Accuracy')
ax.set_title('rte - Accuracy vs Bits')
ax.legend()
ax.grid(True)

# Plot for mrpc
ax = axs[1, 1]
df_filtered = df[(df['Dataset'] == 'mrpc') & (df['Metric'] == 'F1')]
for approach in df_filtered['Approach'].unique():
    df_approach = df_filtered[df_filtered['Approach'] == approach]
    ax.scatter(df_approach['Bits'], df_approach['Value'], label=approach)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits')
ax.set_ylabel('F1')
ax.set_title('mrpc - F1 vs Bits')
ax.legend()
ax.grid(True)
# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('results/mask_values_nan/plot.png')
print("Plots have been saved to results/mask_values_nan/plot.png")
