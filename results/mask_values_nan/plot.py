import matplotlib.pyplot as plt
import pandas as pd

# Data for I-BERT with and without bias shift
data = [
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 5.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 6.0, 'Accuracy': 0.9036697247706422},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 7.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Accuracy': 0.9036697247706422},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 5.0, 'Mcc': 0.0},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 6.0, 'Mcc': 0.51728018358102},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 7.0, 'Mcc': 0.5286324175580216},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 8.0, 'Mcc': 0.5110339437874821},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 5.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 6.0, 'Accuracy': 0.5703971119133574},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 7.0, 'Accuracy': 0.6389891696750902},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 8.0, 'Accuracy': 0.631768953068592},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 5.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 6.0, 'F1': 0.7412008281573499},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 7.0, 'F1': 0.8850174216027874},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 8.0, 'F1': 0.8752079866888519},
]

# Data for I-BERT with and without mask
shift_data = [
    {'Dataset': 'sst2', 'Mode': 'I-BERT_NAN', 'Bits': 4.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERT_NAN', 'Bits': 4.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERT_NAN', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT_NAN', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT_NAN', 'Bits': 5.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERT_NAN', 'Bits': 5.0, 'Mcc': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERT_NAN', 'Bits': 5.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT_NAN', 'Bits': 5.0, 'F1': 0.0},
    {'Dataset': 'sst2', 'Mode': 'I-BERT_NAN', 'Bits': 6.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'cola', 'Mode': 'I-BERT_NAN', 'Bits': 6.0, 'Mcc': 0.5342661861783563},
    {'Dataset': 'rte', 'Mode': 'I-BERT_NAN', 'Bits': 6.0, 'Accuracy': 0.592057761732852},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT_NAN', 'Bits': 6.0, 'F1': 0.8055009823182712},
    {'Dataset': 'sst2', 'Mode': 'I-BERT_NAN', 'Bits': 7.0, 'Accuracy': 0.9013761467889908},
    {'Dataset': 'cola', 'Mode': 'I-BERT_NAN', 'Bits': 7.0, 'Mcc': 0.5266648955454677},
    {'Dataset': 'rte', 'Mode': 'I-BERT_NAN', 'Bits': 7.0, 'Accuracy': 0.6498194945848376},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT_NAN', 'Bits': 7.0, 'F1': 0.8783068783068783},
    {'Dataset': 'sst2', 'Mode': 'I-BERT_NAN', 'Bits': 8.0, 'Accuracy': 0.9025229357798165},
    {'Dataset': 'cola', 'Mode': 'I-BERT_NAN', 'Bits': 8.0, 'Mcc': 0.5474865115851942},
    {'Dataset': 'rte', 'Mode': 'I-BERT_NAN', 'Bits': 8.0, 'Accuracy': 0.6028880866425993},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT_NAN', 'Bits': 8.0, 'F1': 0.8844221105527639}
]
color_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']
# Create DataFrames
df = pd.DataFrame(data)
df_shift = pd.DataFrame(shift_data)

# Merge the two DataFrames on Dataset and Bits
df_merged = pd.merge(df, df_shift, on=['Dataset', 'Bits'], suffixes=('', '_NAN'))

# Calculate the difference between I-BERT and I-BERT_NAN for each metric
df_merged['Accuracy_Diff'] = df_merged['Accuracy'] - df_merged['Accuracy_NAN']
df_merged['Mcc_Diff'] = df_merged['Mcc'] - df_merged['Mcc_NAN']
df_merged['F1_Diff'] = df_merged['F1'] - df_merged['F1_NAN']

# Plotting the differences
fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# Plot for sst2 - Accuracy Difference
ax = axs[0, 0]
df_filtered = df_merged[df_merged['Dataset'] == 'sst2']
ax.plot(df_filtered['Bits'], df_filtered['Accuracy_Diff'], marker='o', color=color_palette[0], linewidth=2.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=14)
ax.set_ylabel('Accuracy Difference', fontsize=14)
ax.set_title('sst2 - Accuracy Difference vs Bits', fontsize=18)
ax.axhline(0, color='black', linestyle='--', linewidth=2.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

# Plot for cola - Mcc Difference
ax = axs[0, 1]
df_filtered = df_merged[df_merged['Dataset'] == 'cola']
ax.plot(df_filtered['Bits'], df_filtered['Mcc_Diff'], marker='o', color=color_palette[1], linewidth=2.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=14)
ax.set_ylabel('Mcc Difference', fontsize=14)
ax.set_title('cola - Mcc Difference vs Bits', fontsize=18)
ax.axhline(0, color='black', linestyle='--', linewidth=2.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

# Plot for rte - Accuracy Difference
ax = axs[1, 0]
df_filtered = df_merged[df_merged['Dataset'] == 'rte']
ax.plot(df_filtered['Bits'], df_filtered['Accuracy_Diff'], marker='o', color=color_palette[2], linewidth=2.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=14)
ax.set_ylabel('Accuracy Difference', fontsize=14)
ax.set_title('rte - Accuracy Difference vs Bits', fontsize=18)
ax.axhline(0, color='black', linestyle='--', linewidth=2.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

# Plot for mrpc - F1 Difference
ax = axs[1, 1]
df_filtered = df_merged[df_merged['Dataset'] == 'mrpc']
ax.plot(df_filtered['Bits'], df_filtered['F1_Diff'], marker='o', color=color_palette[3], linewidth=2.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=14)
ax.set_ylabel('F1 Difference', fontsize=14)
ax.set_title('mrpc - F1 Difference vs Bits', fontsize=18)
ax.axhline(0, color='black', linestyle='--', linewidth=2.5)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('results/mask_values_nan/difference_plot.png')
plt.show()

print("Difference plots have been saved to difference_plot.png")