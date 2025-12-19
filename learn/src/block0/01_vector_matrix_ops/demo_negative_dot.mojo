"""
Demo: Negative Dot Products in ML Context

Shows when and why dot products can be negative.
"""

fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    """Compute dot product."""
    if len(a) != len(b):
        raise Error("Shape mismatch")
    var result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


fn main() raises:
    print("\n" + "="*60)
    print("Negative Dot Products in ML")
    print("="*60 + "\n")

    # Example 1: Neural network with negative weights
    print("1. Neural Network Activation (Common!)")
    print("-" * 40)
    var input: List[Float64] = [1.0, 2.0, 3.0]
    var weights: List[Float64] = [-0.5, -1.0, 0.2]
    var activation = dot_product(input, weights)

    print("Input (features):  ", input)
    print("Weights (learned): ", weights)
    print("Raw activation = w · x =", activation)
    print("→ NEGATIVE! This neuron doesn't detect its pattern here.")
    print("ReLU(", activation, ") = 0 (neuron stays silent)")

    # Example 2: Opposite feature vectors
    print("\n2. Opposite Feature Vectors")
    print("-" * 40)
    var positive_sentiment: List[Float64] = [0.8, 0.6, 0.9]
    var negative_sentiment: List[Float64] = [-0.7, -0.5, -0.8]
    var similarity = dot_product(positive_sentiment, negative_sentiment)

    print("'Happy' embedding:    ", positive_sentiment)
    print("'Sad' embedding:      ", negative_sentiment)
    print("Similarity = ", similarity)
    print("→ NEGATIVE! They're opposites (as expected).")

    # Example 3: When it's positive
    print("\n3. Similar Vectors (Positive)")
    print("-" * 40)
    var cat_features: List[Float64] = [0.9, 0.1, 0.8]
    var tiger_features: List[Float64] = [0.7, 0.2, 0.9]
    var cat_tiger_sim = dot_product(cat_features, tiger_features)

    print("'Cat' features:   ", cat_features)
    print("'Tiger' features: ", tiger_features)
    print("Similarity = ", cat_tiger_sim)
    print("→ POSITIVE! Similar animals, similar features.")

    # Example 4: Error gradient (can be negative)
    print("\n4. Error/Gradient (Training)")
    print("-" * 40)
    var predicted: List[Float64] = [0.8, 0.2]
    var actual: List[Float64] = [0.1, 0.9]
    var error_0 = predicted[0] - actual[0]
    var error_1 = predicted[1] - actual[1]

    print("Predicted: ", predicted)
    print("Actual:    ", actual)
    print("Error[0] = ", error_0, "(positive - overshot)")
    print("Error[1] = ", error_1, "(negative - undershot)")
    print("→ Negative error means 'increase this weight'!")

    print("\n" + "="*60)
    print("Key Takeaway:")
    print("="*60)
    print("✅ Negative dot products are NORMAL in ML!")
    print("✅ They mean: opposite direction, not activated, or error signal")
    print("✅ Activation functions (ReLU, softmax) handle the sign")
    print("="*60 + "\n")
