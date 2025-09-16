# Neural Network for MNIST Digit Recognition

A from-scratch implementation of a 3-layer neural network achieving **74.9% accuracy** on MNIST test data.

## 🧠 Network Architecture

Input (784) → Hidden Layer 1 (128 neurons, ReLU) → Hidden Layer 2 (64 neurons, ReLU) → Output (10 neurons, Softmax)


## ⚙️ Key Techniques Implemented

### Weight Initialization
- **He Initialization**: `W ~ N(0, √(2/n_in))` for ReLU layers
- Prevents vanishing/exploding gradients in deep networks

### Regularization
- **L2 Regularization**: Added to weight updates to prevent overfitting
- λ = 0.01 chosen through cross-validation

### Learning Rate Schedule
- **Cosine Annealing**: `α = α₀ * 0.5*(1 + cos(π * epoch/total_epochs))`
- Gradually reduces learning rate for better convergence

### Optimization
- **Hyperparameter Tuning**: Systematic search over learning rates and regularization strengths
- **Cross-Validation**: 3-fold CV used for model selection

## 📊 Performance
- **Training Accuracy**: ~85% (final epoch)
- **Cross-Validation Accuracy**: 66.0% (3-fold average)
- **Test Accuracy**: 74.9% (on unseen data)

## 🚀 Getting Started

```python
# Train the model
W1, b1, W2, b2, W3, b3 = enhanced_gradient_descent(
    X_train, Y_train, 
    iterations=200, 
    initial_alpha=0.015, 
    lambda_reg=0.01
)

# Make predictions
predictions = get_predictions(forward_prop(W1, b1, W2, b2, W3, b3, X_test)[-1])
