"""Tests for neural network components."""

# Direct imports from the nn module
import nn


fn abs(x: Float64) -> Float64:
    """Absolute value."""
    if x < 0:
        return -x
    return x


fn test_relu() raises:
    """Test ReLU activation."""
    print("Testing ReLU...")

    assert_equal(nn.relu(5.0), 5.0, "Positive should pass through")
    assert_equal(nn.relu(-3.0), 0.0, "Negative should be zero")
    assert_equal(nn.relu(0.0), 0.0, "Zero should be zero")

    var vec: List[Float64] = [1.0, -2.0, 3.0, -4.0, 0.0]
    var result = nn.relu_vector(vec)

    assert_equal(result[0], 1.0, "vec[0] should be 1.0")
    assert_equal(result[1], 0.0, "vec[1] should be 0.0")
    assert_equal(result[2], 3.0, "vec[2] should be 3.0")
    assert_equal(result[3], 0.0, "vec[3] should be 0.0")
    assert_equal(result[4], 0.0, "vec[4] should be 0.0")

    print("  ✓ ReLU tests passed")


fn test_sigmoid() raises:
    """Test sigmoid activation."""
    print("Testing sigmoid...")

    var result = nn.sigmoid(0.0)
    assert_close(result, 0.5, 0.001, "sigmoid(0) should be ~0.5")

    var large_positive = nn.sigmoid(10.0)
    assert_true(large_positive > 0.99, "sigmoid(large positive) should be close to 1")

    var large_negative = nn.sigmoid(-10.0)
    assert_true(large_negative < 0.01, "sigmoid(large negative) should be close to 0")

    print("  ✓ Sigmoid tests passed")


fn test_softmax() raises:
    """Test softmax activation."""
    print("Testing softmax...")

    var logits: List[Float64] = [1.0, 2.0, 3.0]
    var probs = nn.softmax(logits)

    # Check sum to 1.0
    var sum_probs: Float64 = 0.0
    for i in range(len(probs)):
        sum_probs += probs[i]

    assert_close(sum_probs, 1.0, 0.001, "Softmax should sum to 1.0")

    # Check that larger logit gives larger probability
    assert_true(probs[2] > probs[1], "Larger logit should give larger prob")
    assert_true(probs[1] > probs[0], "Larger logit should give larger prob")

    print("  ✓ Softmax tests passed")


fn test_mse_loss() raises:
    """Test MSE loss."""
    print("Testing MSE loss...")

    var preds: List[Float64] = [1.0, 2.0, 3.0]
    var targets: List[Float64] = [1.0, 2.0, 3.0]
    var loss = nn.mse_loss(preds, targets)

    assert_equal(loss, 0.0, "Perfect predictions should have zero loss")

    var preds2: List[Float64] = [2.0, 3.0, 4.0]
    var targets2: List[Float64] = [1.0, 2.0, 3.0]
    var loss2 = nn.mse_loss(preds2, targets2)

    assert_equal(loss2, 1.0, "MSE should be 1.0 for diff of 1 everywhere")

    print("  ✓ MSE loss tests passed")


fn test_dense_layer() raises:
    """Test DenseLayer forward pass."""
    print("Testing DenseLayer...")

    # Create a simple 2-input, 2-output layer
    var weights: List[List[Float64]] = [
        [0.5, -0.5],  # Neuron 1
        [0.3, 0.7]    # Neuron 2
    ]
    var biases: List[Float64] = [0.1, -0.2]

    var layer = nn.DenseLayer(weights, biases, "relu")

    # Test forward pass
    var inputs: List[Float64] = [1.0, 2.0]
    var output = layer.forward(inputs)

    assert_equal(len(output), 2, "Output should have 2 elements")

    # Neuron 1: 0.5*1 + -0.5*2 + 0.1 = -0.4 -> ReLU -> 0.0
    # Neuron 2: 0.3*1 + 0.7*2 + -0.2 = 1.5 -> ReLU -> 1.5
    assert_equal(output[0], 0.0, "Neuron 1 output should be 0.0")
    assert_equal(output[1], 1.5, "Neuron 2 output should be 1.5")

    print("  ✓ DenseLayer tests passed")


fn test_dense_layer_sigmoid() raises:
    """Test DenseLayer with sigmoid activation."""
    print("Testing DenseLayer with sigmoid...")

    var weights: List[List[Float64]] = [[1.0, 1.0]]
    var biases: List[Float64] = [0.0]
    var layer = nn.DenseLayer(weights, biases, "sigmoid")

    var inputs: List[Float64] = [0.0, 0.0]
    var output = layer.forward(inputs)

    # sigmoid(0) = 0.5
    assert_close(output[0], 0.5, 0.01, "sigmoid(0) should be ~0.5")

    print("  ✓ DenseLayer sigmoid tests passed")


fn assert_equal(value: Float64, expected: Float64, message: String) raises:
    """Assert that value equals expected."""
    if value != expected:
        raise Error(message)


fn assert_equal(value: Int, expected: Int, message: String) raises:
    """Assert that value equals expected."""
    if value != expected:
        raise Error(message)


fn assert_close(value: Float64, expected: Float64, tolerance: Float64, message: String) raises:
    """Assert that value is close to expected within tolerance."""
    if abs(value - expected) > tolerance:
        raise Error(message)


fn assert_true(condition: Bool, message: String) raises:
    """Assert that condition is true."""
    if not condition:
        raise Error(message)


fn main() raises:
    """Run all neural network tests."""
    print("\n" + "="*50)
    print("Running Neural Network Tests")
    print("="*50 + "\n")

    test_relu()
    test_sigmoid()
    test_softmax()
    test_mse_loss()
    test_dense_layer()
    test_dense_layer_sigmoid()

    print("\n" + "="*50)
    print("✓ All NN tests passed!")
    print("="*50 + "\n")
