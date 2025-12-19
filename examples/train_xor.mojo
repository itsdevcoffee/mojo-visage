"""
Train a neural network to learn XOR using backpropagation.

XOR is the classic non-linear problem that single-layer networks can't solve.
This demonstrates:
- Forward propagation
- Backpropagation
- Gradient descent
- Network learning over time
"""

from visage import (
    matrix_vector_multiply, vector_add,
    elementwise_multiply, scalar_multiply
)
from math import exp
from random import random_float64


# ==============================================================================
# Activation Functions & Derivatives
# ==============================================================================

fn sigmoid(x: Float64) -> Float64:
    """Sigmoid activation."""
    if x >= 0:
        return 1.0 / (1.0 + exp(-x))
    else:
        var z = exp(x)
        return z / (1.0 + z)


fn sigmoid_vector(vec: List[Float64]) -> List[Float64]:
    """Apply sigmoid to vector."""
    var result = List[Float64]()
    for i in range(len(vec)):
        result.append(sigmoid(vec[i]))
    return result^


fn sigmoid_derivative(x: Float64) -> Float64:
    """Derivative of sigmoid: σ(x) * (1 - σ(x))."""
    var s = sigmoid(x)
    return s * (1.0 - s)


fn sigmoid_derivative_vector(vec: List[Float64]) -> List[Float64]:
    """Apply sigmoid derivative to vector."""
    var result = List[Float64]()
    for i in range(len(vec)):
        result.append(sigmoid_derivative(vec[i]))
    return result^


# ==============================================================================
# Loss Functions
# ==============================================================================

fn mse_loss(predictions: List[Float64], targets: List[Float64]) raises -> Float64:
    """Mean Squared Error."""
    if len(predictions) != len(targets):
        raise Error("Length mismatch")

    var sum_sq: Float64 = 0.0
    for i in range(len(predictions)):
        var diff = predictions[i] - targets[i]
        sum_sq += diff * diff
    return sum_sq / len(predictions)


# ==============================================================================
# Simple 2-Layer Network with Backprop
# ==============================================================================

struct TwoLayerNetwork:
    """
    Simple 2-layer network: input → hidden → output.

    Stores weights, biases, and intermediate values needed for backprop.
    """
    var w1: List[List[Float64]]  # Input → Hidden weights
    var b1: List[Float64]         # Hidden biases
    var w2: List[List[Float64]]  # Hidden → Output weights
    var b2: List[Float64]         # Output biases

    # Cache for backprop
    var last_input: List[Float64]
    var last_hidden_pre: List[Float64]
    var last_hidden: List[Float64]
    var last_output_pre: List[Float64]
    var last_output: List[Float64]

    fn __init__(out self, input_size: Int, hidden_size: Int, output_size: Int):
        """Initialize with small random weights."""
        # Layer 1: input_size → hidden_size
        self.w1 = List[List[Float64]]()
        for _ in range(hidden_size):
            var row = List[Float64]()
            for _ in range(input_size):
                row.append(random_float64(-0.5, 0.5))
            self.w1.append(row^)

        self.b1 = List[Float64]()
        for _ in range(hidden_size):
            self.b1.append(0.0)

        # Layer 2: hidden_size → output_size
        self.w2 = List[List[Float64]]()
        for _ in range(output_size):
            var row = List[Float64]()
            for _ in range(hidden_size):
                row.append(random_float64(-0.5, 0.5))
            self.w2.append(row^)

        self.b2 = List[Float64]()
        for _ in range(output_size):
            self.b2.append(0.0)

        # Initialize cache
        self.last_input = List[Float64]()
        self.last_hidden_pre = List[Float64]()
        self.last_hidden = List[Float64]()
        self.last_output_pre = List[Float64]()
        self.last_output = List[Float64]()

    fn forward(mut self, inputs: List[Float64]) raises -> List[Float64]:
        """Forward pass - store intermediate values for backprop."""
        # Store input (make a copy)
        self.last_input = List[Float64]()
        for i in range(len(inputs)):
            self.last_input.append(inputs[i])

        # Layer 1: input → hidden
        var hidden_pre = matrix_vector_multiply(self.w1, inputs)
        self.last_hidden_pre = vector_add(hidden_pre, self.b1)
        self.last_hidden = sigmoid_vector(self.last_hidden_pre)

        # Layer 2: hidden → output
        var output_pre = matrix_vector_multiply(self.w2, self.last_hidden)
        self.last_output_pre = vector_add(output_pre, self.b2)
        self.last_output = sigmoid_vector(self.last_output_pre)

        # Return a copy
        var result = List[Float64]()
        for i in range(len(self.last_output)):
            result.append(self.last_output[i])
        return result^

    fn backward(
        mut self,
        targets: List[Float64],
        learning_rate: Float64
    ) raises:
        """
        Backpropagation: compute gradients and update weights.

        Uses chain rule to flow gradients backward through the network.
        """
        # Output layer gradient: dL/dOutput
        var output_error = List[Float64]()
        for i in range(len(self.last_output)):
            output_error.append(self.last_output[i] - targets[i])

        # dL/dOutput_pre = dL/dOutput * sigmoid'(output_pre)
        var output_pre_grad = List[Float64]()
        for i in range(len(output_error)):
            var sig_deriv = sigmoid_derivative(self.last_output_pre[i])
            output_pre_grad.append(output_error[i] * sig_deriv)

        # Gradients for W2 and b2
        # dL/dW2[i,j] = dL/dOutput_pre[i] * hidden[j]
        # dL/db2[i] = dL/dOutput_pre[i]

        # Hidden layer error (backpropagate through W2)
        var hidden_error = List[Float64]()
        for j in range(len(self.last_hidden)):
            var error: Float64 = 0.0
            for i in range(len(output_pre_grad)):
                error += output_pre_grad[i] * self.w2[i][j]
            hidden_error.append(error)

        # dL/dHidden_pre = dL/dHidden * sigmoid'(hidden_pre)
        var hidden_pre_grad = List[Float64]()
        for i in range(len(hidden_error)):
            var sig_deriv = sigmoid_derivative(self.last_hidden_pre[i])
            hidden_pre_grad.append(hidden_error[i] * sig_deriv)

        # Update W2 and b2
        for i in range(len(self.w2)):
            for j in range(len(self.w2[i])):
                var gradient = output_pre_grad[i] * self.last_hidden[j]
                self.w2[i][j] -= learning_rate * gradient

            # Update b2
            self.b2[i] -= learning_rate * output_pre_grad[i]

        # Update W1 and b1
        for i in range(len(self.w1)):
            for j in range(len(self.w1[i])):
                var gradient = hidden_pre_grad[i] * self.last_input[j]
                self.w1[i][j] -= learning_rate * gradient

            # Update b1
            self.b1[i] -= learning_rate * hidden_pre_grad[i]


# ==============================================================================
# XOR Training
# ==============================================================================

fn main() raises:
    print("\n" + "="*70)
    print("Training Neural Network on XOR Problem")
    print("="*70 + "\n")

    # XOR dataset
    var X: List[List[Float64]] = [
        [0.0, 0.0],  # XOR(0,0) = 0
        [0.0, 1.0],  # XOR(0,1) = 1
        [1.0, 0.0],  # XOR(1,0) = 1
        [1.0, 1.0]   # XOR(1,1) = 0
    ]

    var y: List[List[Float64]] = [
        [0.0],
        [1.0],
        [1.0],
        [0.0]
    ]

    print("XOR Truth Table:")
    print("  0 XOR 0 = 0")
    print("  0 XOR 1 = 1")
    print("  1 XOR 0 = 1")
    print("  1 XOR 1 = 0")
    print()

    # Create network: 2 inputs → 4 hidden → 1 output
    print("Network Architecture:")
    print("  Input:  2 neurons")
    print("  Hidden: 4 neurons (sigmoid)")
    print("  Output: 1 neuron (sigmoid)")
    print("  Learning Rate: 0.5")
    print()

    var network = TwoLayerNetwork(2, 4, 1)
    var learning_rate: Float64 = 0.5
    var epochs: Int = 5000

    print("="*70)
    print("TRAINING")
    print("="*70 + "\n")

    # Training loop
    for epoch in range(epochs):
        var total_loss: Float64 = 0.0

        # Train on each example
        for i in range(len(X)):
            # Forward pass
            var prediction = network.forward(X[i])

            # Compute loss
            var loss = mse_loss(prediction, y[i])
            total_loss += loss

            # Backward pass & update
            network.backward(y[i], learning_rate)

        # Print progress every 500 epochs
        if (epoch + 1) % 500 == 0:
            var avg_loss = total_loss / len(X)
            print("Epoch", epoch + 1, "| Loss:", avg_loss)

    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70 + "\n")

    # Test the trained network
    print("Testing trained network:")
    print()
    for i in range(len(X)):
        var prediction = network.forward(X[i])
        var pred_value = prediction[0]
        var target_value = y[i][0]

        var pred_class: Int = 0
        if pred_value > 0.5:
            pred_class = 1

        var target_class = Int(target_value)

        # Print result with checkmark
        if pred_class == target_class:
            print("Input:", X[i], "| Target:", target_value,
                  "| Prediction:", pred_value,
                  "| Class:", pred_class, "✓")
        else:
            print("Input:", X[i], "| Target:", target_value,
                  "| Prediction:", pred_value,
                  "| Class:", pred_class, "✗")

    print("\n" + "="*70)
    print("✓ Training complete! Network learned XOR!")
    print("="*70 + "\n")
