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
    {'Dataset': 'mrpc', 'Mode': 'ITA', 'Bits': 8, 'True_Accuracy': 0.8662420382165605},
    {'Dataset': 'sst2', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.9059633027522935},
    {'Dataset': 'cola', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.5312319613383731},
    {'Dataset': 'rte', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.5812274368231047},
    {'Dataset': 'mnli', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.7535404992358635},
    {'Dataset': 'mrpc', 'Mode': 'ITA-Partial', 'Bits': 8, 'True_Accuracy': 0.7634408602150538},
    {'Dataset': 'sst2', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.9048165137614679},
    {'Dataset': 'cola', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.5462521859735874},
    {'Dataset': 'rte', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.5451263537906137},
    {'Dataset': 'mnli', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.731768953068592},
    {'Dataset': 'mrpc', 'Mode': 'Softermax', 'Bits': 8, 'True_Accuracy': 0.7575221238938054},
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

# Create subplots
fig, axs = plt.subplots(3, 2, figsize=(20, 20))

datasets = ['sst2', 'cola', 'rte', 'mnli', 'mrpc']
modes = df['Mode'].unique()

# Plot for each dataset
for i, dataset in enumerate(datasets):
    ax = axs[i // 2, i % 2]
    df_filtered = df[df['Dataset'] == dataset]
    
    for mode in modes:
        df_mode = df_filtered[df_filtered['Mode'] == mode]
        if mode == 'I-BERT':
            ax.plot(df_mode['Bits'], df_mode['True_Accuracy'], marker='o', label=f'{mode} True Accuracy')
            ax.plot(df_mode['Bits'], df_mode['Fake_Accuracy'], marker='x', linestyle='--', label=f'{mode} Fake Accuracy')
        else:
            ax.plot(df_mode['Bits'], df_mode['True_Accuracy'], marker='o', label=f'{mode} True Accuracy')

    # Add the baseline
    ax.axhline(y=baseline_values[dataset], color='r', linestyle='--', label='Baseline')
    
    ax.set_xticks(df_filtered['Bits'].unique())
    ax.set_xlabel('Bits')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{dataset} - Accuracy vs Bits')
    ax.legend()
    ax.grid(True)

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('results/accuracy/plot.png')
print("Plots have been saved to variance_analysis_8bits.png")
