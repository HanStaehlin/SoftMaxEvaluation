import numpy as np
import matplotlib.pyplot as plt

# Function to calculate softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Generate inputs
inputs = np.linspace(-2.0, 2.0, num=100)
outputs = softmax(inputs)

# Plotting the inputs and outputs
plt.figure(figsize=(10, 6))
plt.plot(inputs, outputs, label='Softmax Output')
plt.title('Softmax Function Visualization')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.grid(True)
plt.savefig('softmax_visualization.png')
plt.show()

print("Visualization saved as 'softmax_visualization.png'")