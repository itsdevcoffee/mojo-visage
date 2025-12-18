"""
Block 0 - Section 1: Vector/Matrix Operations
Project 2: Dot Product

Goal: The most important operation in ML - multiply and sum.
Foundation of neural networks, attention, and matrix operations.
"""


fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    """
    Compute dot product: a · b = a[0]*b[0] + a[1]*b[1] + ... + a[n]*b[n].

    This is THE most important operation in ML/AI:
    - Neural networks: weighted sum of inputs
    - Attention: similarity between vectors
    - Matrix multiply: many dot products
    - Loss functions: error calculations

    Args:
        a: First vector.
        b: Second vector (must be same length).

    Returns:
        Scalar result (single number).

    Raises:
        Error if vectors have different lengths.

    Example:
        [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32.
    """
    # STEP 1: Shape validation
    if len(a) != len(b):
        raise Error(
            "Shape mismatch: vectors must have same length. "
            + "Got "
            + String(len(a))
            + " and "
            + String(len(b))
        )

    # STEP 2: Initialize accumulator
    var result: Float64 = 0.0

    # STEP 3: Multiply and accumulate (this is the dot product!)
    for i in range(len(a)):
        result += a[i] * b[i]  # Multiply elements, add to sum

    return result


fn main() raises:
    """Demo: Dot product with key mathematical properties."""
    print("\n=== Dot Product Demo ===\n")

    # Basic dot product
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = dot_product(v1, v2)
    print("1. Basic: [1,2,3] · [4,5,6] =", result, "(expected: 32.0)")

    # Orthogonal vectors
    var v3: List[Float64] = [1.0, 0.0]
    var v4: List[Float64] = [0.0, 1.0]
    var result2 = dot_product(v3, v4)
    print("2. Orthogonal: [1,0] · [0,1] =", result2, "(expected: 0.0)")

    # Magnitude squared
    var v5: List[Float64] = [3.0, 4.0]
    var result3 = dot_product(v5, v5)
    print("3. Magnitude²: [3,4] · [3,4] =", result3, "(expected: 25.0)")

    # Opposite vectors (YOUR IMPLEMENTATION - negative dot product!)
    var v6: List[Float64] = [1.0, 0.0]
    var v7: List[Float64] = [-1.0, 0.0]
    var result4 = dot_product(v6, v7)
    print("4. Opposite: [1,0] · [-1,0] =", result4, "(expected: -1.0)")
    print("   → 180° angle, maximum negative!")

    # Obtuse angle (between 90° and 180°)
    var v8: List[Float64] = [1.0, 1.0]
    var v9: List[Float64] = [-1.0, 0.5]
    var result5 = dot_product(v8, v9)
    print("5. Obtuse angle: [1,1] · [-1,0.5] =", result5, "(negative, ~135°)")

    # Neural network scenario (ML context!)
    print("\n--- Neural Network Example ---")
    var neuron_input: List[Float64] = [0.5, 0.8, 0.3]
    var neuron_weights: List[Float64] = [-1.2, -0.5, 0.2]
    var activation = dot_product(neuron_input, neuron_weights)
    print("Input features:  [0.5, 0.8, 0.3]")
    print("Learned weights: [-1.2, -0.5, 0.2]")
    print("Raw activation = w · x =", activation)
    print("→ NEGATIVE! Neuron inhibited (pattern not detected)")

    # Mixed signs
    var v10: List[Float64] = [-2.0, 3.0, -1.0]
    var v11: List[Float64] = [1.0, -4.0, 2.0]
    var result6 = dot_product(v10, v11)
    print("\n6. Mixed signs: [-2,3,-1] · [1,-4,2] =", result6)
    print("   Breakdown: (-2)(1) + (3)(-4) + (-1)(2) = -2 -12 -2 = -16")

    print("\n✓ Dot product is the foundation of neural networks!")
    print("✓ Negative values are NORMAL and meaningful in ML!")
