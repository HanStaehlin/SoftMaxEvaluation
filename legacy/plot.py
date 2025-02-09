import matplotlib.pyplot as plt

# Define the data including both configurations
results = {
    'sst2': {
        'bits': [4, 6, 8],
        'Original': [0.9037, 0.895, 0.895],
        'Fake_quant': [0.7913, 0.905, 0.9],
        'True_quant': [0.6720, 0.89, 0.895],
        'Original_Sym': [0.895, 0.895, 0.895],
        'Fake_quant_Sym': [0.505, 0.895, 0.895],
        'True_quant_Sym': [0.505, 0.895, 0.9]
    },
    'cola': {
        'bits': [4, 6, 8],
        'Original': [0.5278, 0.5211, 0.5211],
        'Fake_quant': [-0.0177, 0.3402, 0.3075],
        'True_quant': [0.0587, 0.4455, 0.4871],
        'Original_Sym': [0.5211, 0.5211, 0.5211],
        'Fake_quant_Sym': [0.0, 0.5706, 0.5033],
        'True_quant_Sym': [0.0, 0.5609, 0.5264]
    },
    'rte': {
        'bits': [4, 6, 8],
        'Original': [0.6679, 0.665, 0.665],
        'Fake_quant': [0.5162, 0.61, 0.635],
        'True_quant': [0.4549, 0.6, 0.63],
        'Original_Sym': [0.665, 0.665, 0.665],
        'Fake_quant_Sym': [0.5, 0.63, 0.635],
        'True_quant_Sym': [0.51, 0.57, 0.65]
    },
    'mnli': {
        'bits': [4, 6, 8],
        'Original': [0.8190, 0.815, 0.815],
        'Fake_quant': [0.4549, 0.78, 0.79],
        'True_quant': [0.4847, 0.78, 0.78],
        'Original_Sym': [0.815, 0.815, 0.815],
        'Fake_quant_Sym': [0.41, 0.795, 0.79],
        'True_quant_Sym': [0.41, 0.795, 0.81]
    },
    'mrpc': {
        'bits': [4, 6, 8],
        'Original': [None, 0.8973, 0.8973],
        'Fake_quant': [None, 0.8758, 0.8662],
        'True_quant': [None, 0.0567, 0.8551],
        'Original_Sym': [0.8973, 0.8973, 0.8973],
        'Fake_quant_Sym': [0.0145, 0.8702, 0.8662],
        'True_quant_Sym': [0.0, 0.2065, 0.8551]
    }
}

# Plotting
fig, axs = plt.subplots(3, 2, figsize=(12, 18))  # Adjust the subplot grid if necessary
axs = axs.flatten()

# Iterate over each dataset and plot
for idx, (dataset, data) in enumerate(results.items()):
    bits = data['bits']
    models = ['Original', 'Fake_quant', 'True_quant']
    for model_type in models:
        if data.get(f'{model_type}_Sym'):  # Check if symmetric data is available
            axs[idx].plot(bits, data[f'{model_type}_Sym'], marker='s', linestyle='--', label=f"{model_type} Symmetric")
            axs[idx].plot(bits, data[model_type], marker='o', label=f"{model_type}")
            axs[idx].set_title(f'Performance Metrics for {dataset.upper()}')
            axs[idx].set_xlabel('Quantization Bits')
            axs[idx].set_ylabel('Metric Value')
            axs[idx].legend()
            axs[idx].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()
plt.savefig('performance_metrics.png')
# Show the plot
plt.show()