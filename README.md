# Neural Network for MNIST Digit Recognition

A from-scratch implementation of a 3-layer neural network achieving **74.9% accuracy** on MNIST test data.

## Theoretical Foundation
Based on the principles outlined in the [Neural Network Design Document](./NW1.pdf):
- Forward and Backward Propagation
- Weight Initialization strategies
- Learning Rate scheduling
- Early Stopping techniques

## Network Architecture
Input (784) → Hidden Layer 1 (128 neurons, ReLU) → Hidden Layer 2 (64 neurons, ReLU) → Output (10 neurons, Softmax)

## ⚙️ Key Techniques
### Weight Initialization (Page 8)
- He Initialization: W ~ N(0, √(2/n_in)) for ReLU layers

### Learning Rate Schedule (Page 9-10)
- Cosine Annealing: α = α₀ * 0.5*(1 + cos(π * epoch/total_epochs))

### Early Stopping (Page 11)
- Monitor validation loss to prevent overfitting

## Performance
- Test Accuracy: 74.9%
- Cross-Validation: 66.0%

