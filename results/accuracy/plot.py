import matplotlib.pyplot as plt
import pandas as pd

# Data for I-BERT and other methods on 8 bits for different datasets
data = [
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 8, 'Fake_Accuracy': 0.9059633027522935, 'True_Accuracy': 0.9036697247706422},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 8, 'Fake_Accuracy': 0.5338312012443657, 'True_Accuracy': 0.5110339437874821},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 8, 'Fake_Accuracy': 0.5992779783393501, 'True_Accuracy': 0.631768953068592},
    {'Dataset': 'mnli', 'Mode': 'I-BERT', 'Bits': 8, 'Fake_Accuracy': 0.7760570555272542, 'True_Accuracy': 0.8037697401935813},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 8, 'Fake_Accuracy': 0.85625, 'True_Accuracy': 0.8752079866888519},
    {'Dataset': 'sst2', 'Mode': 'ITA', 'Bits': 8, 'True_Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'ITA', 'Bits': 8, 'True_Accuracy': 0.5552324357295336},
    {'Dataset': 'rte', 'Mode': 'ITA', 'Bits': 8, 'True_Accuracy': 0.6209386281588448},
    {'Dataset': 'mnli', 'Mode': 'ITA', 'Bits': 8, 'True_Accuracy': 0.7857361181864493},
    {'Dataset': 'mrpc', 'Mode': 'ITA', 'True_Accuracy': 0.8662420382165605},
    {'Dataset': 'sst2', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.5312319613383731},
    {'Dataset': 'rte', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.5812274368231047},
    {'Dataset': 'mnli', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.7535404992358635},
    {'Dataset': 'mrpc', 'Mode': 'ITA-Partial', 'True_Accuracy': 0.7634408602150538},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.9048165137614679},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.5462521859735874},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.5451263537906137},
    {'Dataset': 'mnli', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.731768953068592},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'True_Accuracy': 0.7575221238938054},
]

# Create a DataFrame
df = pd.DataFrame(data)

# Baseline values
baseline_values = {
    'sst2': 0.9036697247706422,
    'cola': 0.5277813760438573,
    'rte': 0.6678700361010831,
    'mnli': 0.8189505858380031,
    'mrpc': 0.8972602739726028
}

# Random values
random_values = {
    'sst2': 0.4908256880733945,
    'cola': 0.0,
    'rte': 0.5270758122743683,
    'mnli': 0.5,
    'mrpc': 0.0,
}

# Create individual bar plots for each dataset
fig, axs = plt.subplots(1, 5, figsize=(25, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

datasets = ['sst2', 'cola', 'rte', 'mnli', 'mrpc']
modes = df['Mode'].unique()
handles = []

for i, dataset in enumerate(datasets):
    ax = axs[i % 5]
    df_filtered = df[df['Dataset'] == dataset]
    
    bar_width = 0.35
    positions = list(range(len(modes)))
    
    for j, mode in enumerate(modes):
        random_value = random_values[dataset]
        if mode == '':
            true_accuracy = df_filtered[df_filtered['Mode'] == mode]['True_Accuracy'].values[0]
            fake_accuracy = df_filtered[df_filtered['Mode'] == mode]['Fake_Accuracy'].values[0]
            bars1 = ax.bar(positions[j] - bar_width/2, true_accuracy - random_value, bottom=random_value, width=bar_width, label=f'{mode} True Accuracy')
            bars2 = ax.bar(positions[j] + bar_width/2, fake_accuracy - random_value, bottom=random_value, width=bar_width, label=f'{mode} Fake Accuracy', hatch='//')
            if i == 0:
                handles.append(bars1)
                handles.append(bars2)
        else:
            true_accuracy = df_filtered[df_filtered['Mode'] == mode]['True_Accuracy'].values[0]
            bars = ax.bar(positions[j], true_accuracy - random_value, bottom=random_value, width=bar_width, label=f'{mode} True Accuracy')
            if i == 0:
                handles.append(bars)
    
    # Add the baseline
    ax.axhline(y=baseline_values[dataset], color='r', linestyle='--', label='Baseline')
    if i == 0:
        handles.append(ax.axhline(y=baseline_values[dataset], color='r', linestyle='--', label='Baseline'))
    ax.axhline(y=random_values[dataset], color='black', linestyle='--', label='Random')
    if i == 0:
        handles.append(ax.axhline(y=random_values[dataset], color='black', linestyle='--', label='Random'))
    
    ax.set_xticks([pos for pos in positions])
    ax.set_xticklabels(modes, fontsize=18)
    ax.set_xlabel('Modes', fontsize=20)
    ax.set_ylabel('Accuracy', fontsize=20)
    ax.set_title(f'{dataset} - Accuracy', fontsize=22)
    ax.grid(True)
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # Adjust y-axis limit for mnli
    if dataset == 'mnli':
        ax.set_ylim([random_values[dataset], baseline_values[dataset] + 0.02])

# Add a legend
fig.legend(handles=handles, loc='center', bbox_to_anchor=(0.5, 0.05), ncol=6, fontsize=18)

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.savefig('results/accuracy/plot_8bits.png')
print("Plots have been saved to results/accuracy/plot_8bits.png")
plt.show()