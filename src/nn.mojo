"""
Visage ML - Neural Network Components

Activation functions, loss functions, and layer implementations.
"""

from math import exp, tanh
from visage import dot_product, vector_add, scalar_multiply


# ==============================================================================
# Activation Functions
# ==============================================================================

fn relu(x: Float64) -> Float64:
    """
    ReLU activation: max(0, x)

    Most popular activation for hidden layers.
    - Simple and fast
    - Helps avoid vanishing gradients
    - Introduces non-linearity

    Args:
        x: Input value

    Returns:
        max(0, x)
    """
    if x > 0:
        return x
    return 0.0


fn relu_vector(x: List[Float64]) -> List[Float64]:
    """Apply ReLU element-wise to vector."""
    var result = List[Float64]()
    for i in range(len(x)):
        result.append(relu(x[i]))
    return result^


fn sigmoid(x: Float64) -> Float64:
    """
    Sigmoid activation: 1 / (1 + e^(-x))

    Squashes input to (0, 1) range.
    - Used for binary classification
    - Output interpretable as probability
    - Can suffer from vanishing gradients

    Args:
        x: Input value

    Returns:
        Value in (0, 1)
    """
    # Numerically stable sigmoid
    if x >= 0:
        var z = exp(-x)
        return 1.0 / (1.0 + z)
    else:
        var z = exp(x)
        return z / (1.0 + z)


fn sigmoid_vector(x: List[Float64]) -> List[Float64]:
    """Apply sigmoid element-wise to vector."""
    var result = List[Float64]()
    for i in range(len(x)):
        result.append(sigmoid(x[i]))
    return result^


fn tanh_activation(x: Float64) -> Float64:
    """
    Tanh activation: (e^x - e^(-x)) / (e^x + e^(-x))

    Squashes input to (-1, 1) range.
    - Zero-centered (unlike sigmoid)
    - Used in RNNs and some hidden layers
    - Still can have vanishing gradients

    Args:
        x: Input value

    Returns:
        Value in (-1, 1)
    """
    return tanh(x)


fn tanh_vector(x: List[Float64]) -> List[Float64]:
    """Apply tanh element-wise to vector."""
    var result = List[Float64]()
    for i in range(len(x)):
        result.append(tanh_activation(x[i]))
    return result^


fn softmax(x: List[Float64]) -> List[Float64]:
    """
    Softmax activation: converts logits to probabilities.

    exp(x[i]) / sum(exp(x[j]) for all j)

    - Used for multi-class classification
    - Outputs sum to 1.0
    - Each output is a probability

    Args:
        x: Input logits

    Returns:
        Probability distribution
    """
    # Find max for numerical stability
    var max_val = x[0]
    for i in range(1, len(x)):
        if x[i] > max_val:
            max_val = x[i]

    # Compute exp(x - max) for stability
    var exp_values = List[Float64]()
    var sum_exp: Float64 = 0.0

    for i in range(len(x)):
        var exp_val = exp(x[i] - max_val)
        exp_values.append(exp_val)
        sum_exp += exp_val

    # Normalize
    var result = List[Float64]()
    for i in range(len(exp_values)):
        result.append(exp_values[i] / sum_exp)

    return result^


# ==============================================================================
# Activation Derivatives (for Backpropagation)
# ==============================================================================

fn relu_derivative(x: Float64) -> Float64:
    """
    Derivative of ReLU: 1 if x > 0, else 0

    Used in backpropagation to compute gradients.

    Args:
        x: Original input to ReLU

    Returns:
        Gradient: 1.0 if x > 0, else 0.0
    """
    if x > 0:
        return 1.0
    return 0.0


fn relu_derivative_vector(x: List[Float64]) -> List[Float64]:
    """Apply ReLU derivative element-wise."""
    var result = List[Float64]()
    for i in range(len(x)):
        result.append(relu_derivative(x[i]))
    return result^


fn sigmoid_derivative(x: Float64) -> Float64:
    """
    Derivative of sigmoid: σ(x) * (1 - σ(x))

    Args:
        x: Original input to sigmoid

    Returns:
        Gradient
    """
    var s = sigmoid(x)
    return s * (1.0 - s)


fn sigmoid_derivative_vector(x: List[Float64]) -> List[Float64]:
    """Apply sigmoid derivative element-wise."""
    var result = List[Float64]()
    for i in range(len(x)):
        result.append(sigmoid_derivative(x[i]))
    return result^


fn tanh_derivative(x: Float64) -> Float64:
    """
    Derivative of tanh: 1 - tanh²(x)

    Args:
        x: Original input to tanh

    Returns:
        Gradient
    """
    var t = tanh(x)
    return 1.0 - (t * t)


fn tanh_derivative_vector(x: List[Float64]) -> List[Float64]:
    """Apply tanh derivative element-wise."""
    var result = List[Float64]()
    for i in range(len(x)):
        result.append(tanh_derivative(x[i]))
    return result^


# ==============================================================================
# Loss Functions
# ==============================================================================

fn mse_loss(predictions: List[Float64], targets: List[Float64]) raises -> Float64:
    """
    Mean Squared Error loss: (1/n) * Σ(pred - target)²

    Used for regression tasks.

    Args:
        predictions: Model predictions
        targets: True values

    Returns:
        Average squared error

    Raises:
        Error if lengths don't match
    """
    if len(predictions) != len(targets):
        raise Error("Predictions and targets must have same length")

    var sum_squared_error: Float64 = 0.0
    for i in range(len(predictions)):
        var diff = predictions[i] - targets[i]
        sum_squared_error += diff * diff

    return sum_squared_error / len(predictions)


fn binary_cross_entropy(
    predictions: List[Float64],
    targets: List[Float64]
) raises -> Float64:
    """
    Binary Cross-Entropy loss: -Σ(y*log(p) + (1-y)*log(1-p))

    Used for binary classification.

    Args:
        predictions: Predicted probabilities (0 to 1)
        targets: True labels (0 or 1)

    Returns:
        Cross-entropy loss

    Raises:
        Error if lengths don't match
    """
    if len(predictions) != len(targets):
        raise Error("Predictions and targets must have same length")

    var loss: Float64 = 0.0
    var epsilon: Float64 = 1e-7  # For numerical stability

    for i in range(len(predictions)):
        var p = predictions[i]
        var y = targets[i]

        # Clip predictions to avoid log(0)
        if p < epsilon:
            p = epsilon
        elif p > 1.0 - epsilon:
            p = 1.0 - epsilon

        loss += -(y * exp(p) + (1.0 - y) * exp(1.0 - p))

    return loss / len(predictions)


# ==============================================================================
# Loss Derivatives (for Backpropagation)
# ==============================================================================

fn mse_loss_derivative(
    predictions: List[Float64],
    targets: List[Float64]
) raises -> List[Float64]:
    """
    Derivative of MSE loss: 2/n * (predictions - targets)

    Used to start backpropagation from the loss.

    Args:
        predictions: Model outputs
        targets: True values

    Returns:
        Gradient of loss w.r.t. predictions

    Raises:
        Error if lengths don't match
    """
    if len(predictions) != len(targets):
        raise Error("Predictions and targets must have same length")

    var gradient = List[Float64]()
    var n = Float64(len(predictions))

    for i in range(len(predictions)):
        var grad = (2.0 / n) * (predictions[i] - targets[i])
        gradient.append(grad)

    return gradient^


# ==============================================================================
# Dense Layer
# ==============================================================================

struct DenseLayer:
    """
    Fully-connected (dense) neural network layer.

    Performs: output = activation(weights @ input + bias)

    This is the fundamental building block of neural networks.
    """
    var weights: List[List[Float64]]  # Shape: (output_size, input_size)
    var biases: List[Float64]         # Shape: (output_size,)
    var activation: String            # "relu", "sigmoid", "tanh", or "none"

    fn __init__(
        out self,
        weights: List[List[Float64]],
        biases: List[Float64],
        activation: String = "relu"
    ):
        """
        Initialize dense layer.

        Args:
            weights: Weight matrix (output_size × input_size)
            biases: Bias vector (output_size,)
            activation: Activation function name
        """
        self.weights = weights
        self.biases = biases
        self.activation = activation

    fn forward(self, inputs: List[Float64]) raises -> List[Float64]:
        """
        Forward pass through the layer.

        Args:
            inputs: Input vector (input_size,)

        Returns:
            Output vector (output_size,) after activation

        Raises:
            Error if input size doesn't match weight dimensions
        """
        # Compute weighted sum: W @ x
        var weighted_sum = List[Float64]()

        for i in range(len(self.weights)):
            var neuron_output = dot_product(self.weights[i], inputs)
            weighted_sum.append(neuron_output)

        # Add bias: W @ x + b
        var pre_activation = vector_add(weighted_sum, self.biases)

        # Apply activation function
        if self.activation == "relu":
            return relu_vector(pre_activation^)
        elif self.activation == "sigmoid":
            return sigmoid_vector(pre_activation^)
        elif self.activation == "tanh":
            return tanh_vector(pre_activation^)
        elif self.activation == "softmax":
            return softmax(pre_activation^)
        elif self.activation == "none":
            return pre_activation^
        else:
            raise Error("Unknown activation: " + self.activation)
