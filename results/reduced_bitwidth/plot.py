import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data from the image file
data = [
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 4.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 5.0, 'Accuracy': 0.7660550458715596},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 6.0, 'Accuracy': 0.9036697247706422},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 7.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Accuracy': 0.9036697247706422},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 4.0, 'Mcc': 0.0},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 5.0, 'Mcc': 0.35388748299431894},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 6.0, 'Mcc': 0.51728018358102},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 7.0, 'Mcc': 0.5286324175580216},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 8.0, 'Mcc': 0.5110339437874821},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 5.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 6.0, 'Accuracy': 0.5703971119133574},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 7.0, 'Accuracy': 0.6389891696750902},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 8.0, 'Accuracy': 0.631768953068592},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 5.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 6.0, 'F1': 0.7412008281573499},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 7.0, 'F1': 0.8850174216027874},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 8.0, 'F1': 0.8752079866888519},
    {'Dataset': 'sst2', 'Mode': 'ITA', 'Bits': 4.0, 'Accuracy': 0.573394495412844},
    {'Dataset': 'sst2', 'Mode': 'ITA', 'Bits': 5.0, 'Accuracy': 0.8176605504587156},
    {'Dataset': 'sst2', 'Mode': 'ITA', 'Bits': 6.0, 'Accuracy': 0.9036697247706422},
    {'Dataset': 'sst2', 'Mode': 'ITA', 'Bits': 7.0, 'Accuracy': 0.9094036697247706},
    {'Dataset': 'sst2', 'Mode': 'ITA', 'Bits': 8.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'ITA', 'Bits': 4.0, 'Mcc': -0.028044982189654497},
    {'Dataset': 'cola', 'Mode': 'ITA', 'Bits': 5.0, 'Mcc': 0.5222017375430389},
    {'Dataset': 'cola', 'Mode': 'ITA', 'Bits': 6.0, 'Mcc': 0.5241981070980204},
    {'Dataset': 'cola', 'Mode': 'ITA', 'Bits': 7.0, 'Mcc': 0.5290831606897504},
    {'Dataset': 'cola', 'Mode': 'ITA', 'Bits': 8.0, 'Mcc': 0.5552324357295336},
    {'Dataset': 'rte', 'Mode': 'ITA', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'ITA', 'Bits': 5.0, 'Accuracy': 0.555956678700361},
    {'Dataset': 'rte', 'Mode': 'ITA', 'Bits': 6.0, 'Accuracy': 0.6137184115523465},
    {'Dataset': 'rte', 'Mode': 'ITA', 'Bits': 7.0, 'Accuracy': 0.6534296028880866},
    {'Dataset': 'rte', 'Mode': 'ITA', 'Bits': 8.0, 'Accuracy': 0.6209386281588448},
    {'Dataset': 'mrpc', 'Mode': 'ITA', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'ITA', 'Bits': 5.0, 'F1': 0.19558359621451105},
    {'Dataset': 'mrpc', 'Mode': 'ITA', 'Bits': 6.0, 'F1': 0.855098389982111},
    {'Dataset': 'mrpc', 'Mode': 'ITA', 'Bits': 7.0, 'F1': 0.8811881188118812},
    {'Dataset': 'mrpc', 'Mode': 'ITA', 'Bits': 8.0, 'F1': 0.8662420382165605},
    {'Dataset': 'sst2', 'Mode': 'ITA-Partial', 'Bits': 4.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'ITA-Partial', 'Bits': 5.0, 'Accuracy': 0.8165137614678899},
    {'Dataset': 'sst2', 'Mode': 'ITA-Partial', 'Bits': 6.0, 'Accuracy': 0.9025229357798165},
    {'Dataset': 'sst2', 'Mode': 'ITA-Partial', 'Bits': 7.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'sst2', 'Mode': 'ITA-Partial', 'Bits': 8.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'ITA-Partial', 'Bits': 4.0, 'Mcc': -0.028044982189654497},
    {'Dataset': 'cola', 'Mode': 'ITA-Partial', 'Bits': 5.0, 'Mcc': 0.49702553536108757},
    {'Dataset': 'cola', 'Mode': 'ITA-Partial', 'Bits': 6.0, 'Mcc': 0.5138995234247261},
    {'Dataset': 'cola', 'Mode': 'ITA-Partial', 'Bits': 7.0, 'Mcc': 0.5417526808280421},
    {'Dataset': 'cola', 'Mode': 'ITA-Partial', 'Bits': 8.0, 'Mcc': 0.5312319613383731},
    {'Dataset': 'rte', 'Mode': 'ITA-Partial', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'ITA-Partial', 'Bits': 5.0, 'Accuracy': 0.5631768953068592},
    {'Dataset': 'rte', 'Mode': 'ITA-Partial', 'Bits': 6.0, 'Accuracy': 0.592057761732852},
    {'Dataset': 'rte', 'Mode': 'ITA-Partial', 'Bits': 7.0, 'Accuracy': 0.5956678700361011},
    {'Dataset': 'mrpc', 'Mode': 'ITA-Partial', 'Bits': 4.0, 'F1': -0.028044982189654497},
    {'Dataset': 'mrpc', 'Mode': 'ITA-Partial', 'Bits': 5.0, 'F1': 0.2598187311178248},
    {'Dataset': 'mrpc', 'Mode': 'ITA-Partial', 'Bits': 6.0, 'F1': 0.7733333333333333},
    {'Dataset': 'mrpc', 'Mode': 'ITA-Partial', 'Bits': 7.0, 'F1': 0.7757352941176471},
    {'Dataset': 'rte', 'Mode': 'ITA-Partial', 'Bits': 8.0, 'Accuracy': 0.5812274368231047},
    {'Dataset': 'mrpc', 'Mode': 'ITA-Partial', 'Bits': 8.0, 'F1': 0.7634408602150538},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 4.0, 'Accuracy': 0.4908256880733945},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 5.0, 'Accuracy': 0.768348623853211},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 6.0, 'Accuracy': 0.9059633027522935},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 7.0, 'Accuracy': 0.9013761467889908},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 8.0, 'Accuracy': 0.9048165137614679},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 4.0, 'Mcc': -0.028044982189654497},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 5.0, 'Mcc': 0.4491213009578182},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 6.0, 'Mcc': 0.5244371898114317},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 7.0, 'Mcc': 0.5330482808147131},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 8.0, 'Mcc': 0.5462521859735874},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 4.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 5.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 6.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 7.0, 'Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 4.0, 'F1': 0.0},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 5.0, 'F1': 0.014084507042253521},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 6.0, 'F1': 0.6414414414414414},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 7.0, 'F1': 0.7958271236959762},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 8.0, 'Accuracy': 0.5451263537906137},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 8.0, 'F1': 0.7575221238938054},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Baseline values
baseline_values = {
    'sst2': 0.9036697247706422,
    'cola': 0.5277813760438573,
    'rte': 0.6678700361010831,
    'mrpc': 0.8972602739726028
}

# Random values
random_values = {
    'sst2': 0.4908256880733945,
    'cola': 0.0,
    'rte': 0.5270758122743683,
    'mrpc': 0.0
}

# Function to ensure all bits are present with additional data included
def ensure_all_bits(df, dataset, metric):
    bits = [4.0, 5.0, 6.0, 7.0, 8.0]
    modes = df['Mode'].unique()
    complete_data = []
    for mode in modes:
        df_mode = df[(df['Dataset'] == dataset) & (df['Mode'] == mode)][['Bits', metric]]
        df_mode.set_index('Bits', inplace=True)
        df_mode = df_mode.reindex(bits)
        df_mode['Mode'] = mode
        df_mode['Dataset'] = dataset
        complete_data.append(df_mode.reset_index())
    return pd.concat(complete_data)

# Colorblind-friendly color palette
color_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(30, 20))  # Increase figure size for presentation

# Plot for sst2
ax = axs[0, 0]
df_filtered = ensure_all_bits(df, 'sst2', 'Accuracy')
for i, mode in enumerate(df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    ax.plot(df_mode['Bits'], df_mode['Accuracy'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=3.5, linestyle= None)
ax.axhline(y=baseline_values['sst2'], color='r', linestyle='--', label='Original Model', linewidth=3.5, alpha=0.5)
ax.axhline(y=random_values['sst2'], color='black', linestyle='--', label='Random Guess', linewidth=3.5, alpha = 0.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=24)
ax.set_ylabel('Accuracy', fontsize=24)
ax.set_title('sst2 - Accuracy vs Bits', fontsize=28)
ax.legend(fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True)

# Plot for cola
ax = axs[0, 1]
df_filtered = ensure_all_bits(df, 'cola', 'Mcc')
for i, mode in enumerate(df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    ax.plot(df_mode['Bits'], df_mode['Mcc'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=3.5)
ax.axhline(y=baseline_values['cola'], color='r', linestyle='--', label='Original Model', linewidth=3.5)
ax.axhline(y=random_values['cola'], color='black', linestyle='--', label='Random Guess', linewidth=3.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=24)
ax.set_ylabel('Mcc', fontsize=24)
ax.set_title('cola - Mcc vs Bits', fontsize=28)
ax.legend(fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True)

# Plot for rte
ax = axs[1, 0]
df_filtered = ensure_all_bits(df, 'rte', 'Accuracy')
for i, mode in enumerate(df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    ax.plot(df_mode['Bits'], df_mode['Accuracy'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=3.5)
ax.axhline(y=baseline_values['rte'], color='r', linestyle='--', label='Original Model', linewidth=3.5)
ax.axhline(y=random_values['rte'], color='black', linestyle='--', label='Random Guess', linewidth=3.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=24)
ax.set_ylabel('Accuracy', fontsize=24)
ax.set_title('rte - Accuracy vs Bits', fontsize=28)
ax.legend(fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True)

# Plot for mrpc
ax = axs[1, 1]
df_filtered = ensure_all_bits(df, 'mrpc', 'F1')
for i, mode in enumerate (df_filtered['Mode'].unique()):
    df_mode = df_filtered[df_filtered['Mode'] == mode]
    ax.plot(df_mode['Bits'], df_mode['F1'], label=mode, marker='o', color=color_palette[i % len(color_palette)], linewidth=3.5)
ax.axhline(y=baseline_values['mrpc'], color='r', linestyle='--', label='Original Model', linewidth=3.5)
ax.axhline(y=random_values['mrpc'], color='black', linestyle='--', label='Random Guess', linewidth=3.5)
ax.set_xticks(df_filtered['Bits'].unique())
ax.set_xlabel('Bits', fontsize=24)
ax.set_ylabel('F1', fontsize=24)
ax.set_title('mrpc - F1 vs Bits', fontsize=28)
ax.legend(fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid(True)


# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('results/reduced_bitwidth/plot.png')
print("Plots have been saved to results/reduced_bitwidth/plot.png")