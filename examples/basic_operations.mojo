"""
Basic usage examples for Visage ML library.

Demonstrates core linear algebra operations.
"""

from visage import vector_add, dot_product, matrix_vector_multiply


fn main() raises:
    print("\n" + "="*60)
    print("Visage ML - Basic Operations Demo")
    print("="*60 + "\n")

    # Example 1: Vector Addition
    print("1. Vector Addition")
    print("-" * 40)
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var sum = vector_add(v1, v2)
    print("v1:      ", v1)
    print("v2:      ", v2)
    print("v1 + v2: ", sum)
    print()

    # Example 2: Dot Product
    print("2. Dot Product (Vector Similarity)")
    print("-" * 40)
    var a: List[Float64] = [1.0, 2.0, 3.0]
    var b: List[Float64] = [4.0, 5.0, 6.0]
    var dot = dot_product(a, b)
    print("a:     ", a)
    print("b:     ", b)
    print("a · b: ", dot)
    print()

    # Example 3: Matrix-Vector Multiply (Neural Network Layer)
    print("3. Matrix-Vector Multiply (Neural Network Layer)")
    print("-" * 40)
    var weights: List[List[Float64]] = [
        [0.5, -0.2, 0.1],   # Neuron 1 weights
        [0.3, 0.8, -0.4],   # Neuron 2 weights
        [-0.1, 0.6, 0.7]    # Neuron 3 weights
    ]
    var inputs: List[Float64] = [1.0, 2.0, 3.0]

    print("Input features:  ", inputs)
    print("Weight matrix (3 neurons):")
    for i in range(len(weights)):
        print("  Neuron", i+1, ":", weights[i])

    var output = matrix_vector_multiply(weights, inputs)
    print("\nNeuron activations:", output)
    print()

    # Example 4: Perpendicular Vectors
    print("4. Perpendicular Vectors (Orthogonality)")
    print("-" * 40)
    var x_axis: List[Float64] = [1.0, 0.0]
    var y_axis: List[Float64] = [0.0, 1.0]
    var dot_perp = dot_product(x_axis, y_axis)
    print("x-axis: ", x_axis)
    print("y-axis: ", y_axis)
    print("Dot product:", dot_perp, "(perpendicular vectors → 0)")
    print()

    print("="*60)
    print("✓ Demo complete! These are the building blocks of neural networks.")
    print("="*60 + "\n")
