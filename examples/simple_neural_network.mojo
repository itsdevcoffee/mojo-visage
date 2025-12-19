"""
Simple 2-layer neural network example using Visage ML.

Demonstrates:
- Matrix operations
- Activation functions
- Forward propagation through multiple layers
- Loss calculation
"""

from visage import (
    matrix_vector_multiply, vector_add,
    matrix_matrix_multiply, transpose,
    scalar_multiply
)
from math import exp


# ==============================================================================
# Activation Functions
# ==============================================================================

fn relu(x: Float64) -> Float64:
    """ReLU: max(0, x)"""
    if x > 0:
        return x
    return 0.0


fn relu_vector(vec: List[Float64]) -> List[Float64]:
    """Apply ReLU to vector."""
    var result = List[Float64]()
    for i in range(len(vec)):
        result.append(relu(vec[i]))
    return result^


fn sigmoid(x: Float64) -> Float64:
    """Sigmoid: 1 / (1 + e^(-x))"""
    if x >= 0:
        var z = exp(-x)
        return 1.0 / (1.0 + z)
    else:
        var z = exp(x)
        return z / (1.0 + z)


fn sigmoid_vector(vec: List[Float64]) -> List[Float64]:
    """Apply sigmoid to vector."""
    var result = List[Float64]()
    for i in range(len(vec)):
        result.append(sigmoid(vec[i]))
    return result^


# ==============================================================================
# Loss Function
# ==============================================================================

fn mse_loss(predictions: List[Float64], targets: List[Float64]) raises -> Float64:
    """Mean Squared Error loss."""
    if len(predictions) != len(targets):
        raise Error("Length mismatch")

    var sum_squared_error: Float64 = 0.0
    for i in range(len(predictions)):
        var diff = predictions[i] - targets[i]
        sum_squared_error += diff * diff

    return sum_squared_error / len(predictions)


# ==============================================================================
# Neural Network Forward Pass
# ==============================================================================

fn forward_layer(
    inputs: List[Float64],
    weights: List[List[Float64]],
    biases: List[Float64],
    activation: String
) raises -> List[Float64]:
    """
    Forward pass through one layer.

    Args:
        inputs: Input vector
        weights: Weight matrix (output_size × input_size)
        biases: Bias vector
        activation: "relu", "sigmoid", or "none"

    Returns:
        Layer output after activation
    """
    # Weighted sum: W @ x
    var weighted_sum = matrix_vector_multiply(weights, inputs)

    # Add bias: W @ x + b
    var pre_activation = vector_add(weighted_sum, biases)

    # Apply activation
    if activation == "relu":
        return relu_vector(pre_activation^)
    elif activation == "sigmoid":
        return sigmoid_vector(pre_activation^)
    else:
        return pre_activation^


# ==============================================================================
# Main: Train a Simple Network
# ==============================================================================

fn main() raises:
    print("\n" + "="*70)
    print("Simple 2-Layer Neural Network Example")
    print("="*70 + "\n")

    print("Network Architecture:")
    print("  Input:  2 features")
    print("  Hidden: 3 neurons (ReLU)")
    print("  Output: 1 neuron (Sigmoid)")
    print("  Task:   Binary classification\n")

    # Layer 1: 2 inputs → 3 hidden neurons
    var layer1_weights: List[List[Float64]] = [
        [0.5, -0.3],   # Hidden neuron 1
        [0.2, 0.8],    # Hidden neuron 2
        [-0.4, 0.6]    # Hidden neuron 3
    ]
    var layer1_biases: List[Float64] = [0.1, -0.2, 0.3]

    # Layer 2: 3 hidden → 1 output neuron
    var layer2_weights: List[List[Float64]] = [
        [0.7, -0.5, 0.9]  # Output neuron
    ]
    var layer2_biases: List[Float64] = [0.0]

    # Training example
    var input_features: List[Float64] = [1.5, 2.0]
    var target_output: List[Float64] = [1.0]  # True label

    print("="*70)
    print("FORWARD PROPAGATION")
    print("="*70 + "\n")

    # Input
    print("Input features:", input_features)
    print()

    # Layer 1: Hidden layer
    print("Layer 1 (Hidden):")
    print("  Weights (3×2):")
    for i in range(len(layer1_weights)):
        print("    Neuron", i+1, ":", layer1_weights[i])
    print("  Biases:", layer1_biases)

    var hidden = forward_layer(input_features, layer1_weights, layer1_biases, "relu")
    print("  Output (after ReLU):", hidden)
    print()

    # Layer 2: Output layer
    print("Layer 2 (Output):")
    print("  Weights (1×3):", layer2_weights[0])
    print("  Biases:", layer2_biases)

    var output = forward_layer(hidden, layer2_weights, layer2_biases, "sigmoid")
    print("  Output (after Sigmoid):", output)
    print()

    # Loss
    print("="*70)
    print("LOSS CALCULATION")
    print("="*70 + "\n")

    var loss = mse_loss(output, target_output)
    print("Prediction:", output[0])
    print("Target:    ", target_output[0])
    print("Loss (MSE):", loss)
    print()

    # Interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70 + "\n")

    if output[0] > 0.5:
        print("Model predicts: Class 1 (probability:", output[0], ")")
    else:
        print("Model predicts: Class 0 (probability:", 1.0 - output[0], ")")

    if target_output[0] == 1.0:
        print("True label:     Class 1")
    else:
        print("True label:     Class 0")

    if loss < 0.1:
        print("\nPrediction quality: EXCELLENT")
    elif loss < 0.25:
        print("\nPrediction quality: GOOD")
    else:
        print("\nPrediction quality: POOR (needs training!)")

    print("\n" + "="*70)
    print("✓ Forward pass complete!")
    print("="*70 + "\n")

    print("What's happening here:")
    print("  1. Input features pass through hidden layer (3 neurons with ReLU)")
    print("  2. Hidden activations pass through output layer (1 neuron with sigmoid)")
    print("  3. Loss measures how far prediction is from target")
    print("  4. In real training, we'd use backprop to update weights")
    print()
