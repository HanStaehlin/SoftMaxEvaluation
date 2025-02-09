import pandas as pd
import matplotlib.pyplot as plt

# Colorblind-friendly color palette
color_palette = ['#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3', '#999999', '#e41a1c', '#dede00']

# Data for Fake Quantized Values against Percentiles on SST2
data = [
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 100, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.9071100917431193, 'True_Quant_Accuracy': 0.9025229357798165},
    #{'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 100, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.9071100917431193, 'True_Quant_Accuracy': 0.9025229357798165},
    #{'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 99.9, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.9059633027522935, 'True_Quant_Accuracy': 0.9071100917431193},
    #{'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 99, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.8979357798165137, 'True_Quant_Accuracy': 0.9025229357798165},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 95, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.8818807339449541, 'True_Quant_Accuracy': 0.9025229357798165},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 90, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.8543577981651376, 'True_Quant_Accuracy': 0.9002293577981652},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 85, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.8394495412844036, 'True_Quant_Accuracy': 0.9013761467889908},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 80, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.8360091743119266, 'True_Quant_Accuracy': 0.9036697247706422},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 75, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.8211009174311926, 'True_Quant_Accuracy': 0.9013761467889908},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 70, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.8073394495412844, 'True_Quant_Accuracy': 0.8990825688073395},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 65, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.7970183486238532, 'True_Quant_Accuracy': 0.9048165137614679},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 60, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.7889908256880734, 'True_Quant_Accuracy': 0.9036697247706422},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 55, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.7775229357798165, 'True_Quant_Accuracy': 0.9025229357798165},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 50, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.7660550458715596, 'True_Quant_Accuracy': 0.8956422018348624},
    {'Dataset': 'sst2', 'Model': 'Alireza1044/mobilebert_sst2', 'Mode': 'I-BERT', 'Bits': 8.0, 'Clip_Bounds_Sym': True, 'Percentile': 45, 'Original_Accuracy': 'skipped', 'Fake_Quant_Accuracy': 0.7580275229357798, 'True_Quant_Accuracy': 0.8967889908256881}
]

# Create DataFrame
df = pd.DataFrame(data)

# Plot Fake Quantized Values against Percentiles for SST2
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df['Percentile'], df['Fake_Quant_Accuracy'], label='Fake Quant Model', marker='o', color=color_palette[0], linewidth=3.5)
#ax.plot(df['Percentile'], df['True_Quant_Accuracy'], label='True Quant Model', marker='o', color=color_palette[1], linewidth=3.5)
ax.set_xticks(df['Percentile'].unique())
ax.set_xlabel('Percentile', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)
ax.set_title('sst2 - Fake Quantized Values vs Percentiles', fontsize=16)
ax.legend(fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True)

# Show plot
plt.tight_layout()
plt.show()
plt.savefig('results/percentile/plot.png')
print("Plots have been saved to bias_shift_analysis.png")