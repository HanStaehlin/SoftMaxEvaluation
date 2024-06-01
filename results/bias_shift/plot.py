import matplotlib.pyplot as plt
import pandas as pd

# Data for I-BERT with and without bias shift
data = [
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'False', 'Fake_Accuracy': 0.9048165137614679, 'True_Accuracy': 0.7522935779816514},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'False', 'Fake_Accuracy': 0.4715995041922339, 'True_Accuracy': 0.06042327451990865},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'False', 'Fake_Accuracy': 0.5379061371841155, 'True_Accuracy': 0.5415162454873647},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'False', 'Fake_Accuracy': 0.8447653429602888, 'True_Accuracy': 0.08783783783783784},
    {'Dataset': 'sst2', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'True', 'Fake_Accuracy': 0.8474770642201835, 'True_Accuracy': 0.4908256880733945},
    {'Dataset': 'cola', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'True', 'Fake_Accuracy': 0.13794017432389633, 'True_Accuracy': 0.0},
    {'Dataset': 'rte', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'True', 'Fake_Accuracy': 0.5379061371841155, 'True_Accuracy': 0.5270758122743683},
    {'Dataset': 'mrpc', 'Mode': 'I-BERT', 'Bits': 5, 'Clip_Bounds_Sym': 'True', 'Fake_Accuracy': 0.6339468302658486, 'True_Accuracy': 0.0},
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

# Create subplots
fig, axs = plt.subplots(2, 2, figsize=(20, 20))

datasets = ['sst2', 'cola', 'rte', 'mrpc']

# Plot for each dataset
for i, dataset in enumerate(datasets):
    ax = axs[i // 2, i % 2]
    df_filtered = df[df['Dataset'] == dataset]
    
    for clip_bounds_sym in df_filtered['Clip_Bounds_Sym'].unique():
        df_clip = df_filtered[df_filtered['Clip_Bounds_Sym'] == clip_bounds_sym]
        label_suffix = 'Shifted Bias' if clip_bounds_sym == 'False' else 'Original Bias'
        ax.plot(df_clip['Bits'], df_clip['True_Accuracy'], marker='o', label=f'True Accuracy {label_suffix}')
        ax.plot(df_clip['Bits'], df_clip['Fake_Accuracy'], marker='x', linestyle='--', label=f'Fake Accuracy {label_suffix}')

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
plt.savefig('results/bias_shift/plot.png')
print("Plots have been saved to bias_shift_analysis.png")
