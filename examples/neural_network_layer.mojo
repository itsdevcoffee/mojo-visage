"""
Example: Simple neural network layer using Visage ML.

Demonstrates how matrix-vector operations form the foundation
of neural network computation.
"""

from visage import matrix_vector_multiply, dot_product


fn sigmoid(x: Float64) -> Float64:
    """Sigmoid activation function."""
    # Simplified sigmoid for demonstration
    if x >= 0:
        return 1.0 / (1.0 + (-x))  # Approximate
    else:
        return x / (1.0 + abs(x))  # Approximate for negative values


fn apply_activation(values: List[Float64]) -> List[Float64]:
    """Apply sigmoid to each value."""
    var result = List[Float64]()
    for i in range(len(values)):
        result.append(sigmoid(values[i]))
    return result^


fn main() raises:
    print("\n" + "="*60)
    print("Neural Network Layer Example")
    print("="*60 + "\n")

    # Define a simple 3-input, 2-output neural network layer
    print("Network Architecture:")
    print("  Input:  3 features")
    print("  Layer:  2 neurons")
    print("  Output: 2 activations")
    print()

    # Weight matrix: each row is one neuron's weights
    var weights: List[List[Float64]] = [
        [0.8, -0.5, 0.3],   # Neuron 1
        [-0.2, 0.6, 0.9]    # Neuron 2
    ]

    # Bias terms (added after weighted sum)
    var biases: List[Float64] = [0.1, -0.3]

    # Example input
    var input_features: List[Float64] = [1.5, 2.0, -0.5]

    print("Input features:", input_features)
    print()
    print("Weights:")
    print("  Neuron 1:", weights[0], "+ bias", biases[0])
    print("  Neuron 2:", weights[1], "+ bias", biases[1])
    print()

    # Forward pass: W @ x
    var weighted_sum = matrix_vector_multiply(weights, input_features)

    print("Weighted sums (before activation):")
    for i in range(len(weighted_sum)):
        print("  Neuron", i+1, ":", weighted_sum[i], "+", biases[i], "=", weighted_sum[i] + biases[i])

    # Add biases
    var pre_activation = List[Float64]()
    for i in range(len(weighted_sum)):
        pre_activation.append(weighted_sum[i] + biases[i])

    # Apply activation function
    var activations = apply_activation(pre_activation)

    print()
    print("Final activations (after sigmoid):")
    for i in range(len(activations)):
        print("  Neuron", i+1, ":", activations[i])

    print()
    print("="*60)
    print("✓ This is how neural networks process information!")
    print("  Step 1: Weighted sum (matrix multiply)")
    print("  Step 2: Add bias")
    print("  Step 3: Apply activation function")
    print("="*60 + "\n")
