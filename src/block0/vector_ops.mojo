"""
Block 0: Vector Operations.
Build from scratch to understand shapes, memory, and Mojo fundamentals.
"""

# ============================================================================
# LESSON 1: VECTOR ADDITION
# ============================================================================

fn vector_add(a: List[Float64], b: List[Float64]) raises -> List[Float64]:
    """
    Add two vectors element-wise: c[i] = a[i] + b[i].

    Args:
        a: First vector.
        b: Second vector (must be same length as a).

    Returns:
        New vector containing element-wise sum.

    Raises:
        Error if vectors have different lengths (shape mismatch).

    Why this matters:
        Shape bugs are the #1 source of ML errors.
        Always validate shapes before operating.
        Failing fast saves debugging time.
    """
    # STEP 1: Shape validation (catch bugs early!)
    if len(a) != len(b):
        raise Error("Shape mismatch: vectors must have same length. "
                   + "Got " + String(len(a)) + " and " + String(len(b)))

    # STEP 2: Create result vector
    var result = List[Float64]()

    # STEP 3: Element-wise addition
    for i in range(len(a)):
        result.append(a[i] + b[i])

    # STEP 4: Transfer ownership (Mojo memory management)
    return result^


fn print_vector(v: List[Float64], name: String = "vector"):
    """Helper function to print vectors nicely."""
    print(name + ": [", end="")
    for i in range(len(v)):
        print(v[i], end="")
        if i < len(v) - 1:
            print(", ", end="")
    print("]")


# ============================================================================
# LESSON 2: DOT PRODUCT (Foundation of Neural Networks!)
# ============================================================================

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
        [1, 2, 3] · [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32
    """
    # STEP 1: Shape validation (same as vector_add)
    if len(a) != len(b):
        raise Error("Shape mismatch: vectors must have same length. "
                   + "Got " + String(len(a)) + " and " + String(len(b)))

    # STEP 2: Initialize accumulator
    var result: Float64 = 0.0

    # STEP 3: Multiply and accumulate (this is the dot product!)
    for i in range(len(a)):
        result += a[i] * b[i]  # Multiply elements, add to sum

    return result


# ============================================================================
# TEST: Vector Addition
# ============================================================================

fn test_vector_add() raises:
    """Test our vector addition implementation."""
    print("\n" + "="*60)
    print("Testing Vector Addition")
    print("="*60)

    # Test 1: Basic addition
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = vector_add(v1, v2)

    print_vector(v1, "v1")
    print_vector(v2, "v2")
    print_vector(result, "v1 + v2")
    print("Expected: [5.0, 7.0, 9.0]")

    # Test 2: Adding zeros (identity)
    print("\n--- Test 2: Identity (adding zeros) ---")
    var v3: List[Float64] = [1.0, 2.0, 3.0]
    var zeros: List[Float64] = [0.0, 0.0, 0.0]
    var result2 = vector_add(v3, zeros)
    print_vector(v3, "v3")
    print_vector(zeros, "zeros")
    print_vector(result2, "v3 + zeros")
    print("Expected: [1.0, 2.0, 3.0] (same as v3)")

    # Test 3: Negative numbers
    print("\n--- Test 3: Negative numbers ---")
    var v4: List[Float64] = [5.0, -3.0, 2.0]
    var v5: List[Float64] = [-1.0, 3.0, -2.0]
    var result3 = vector_add(v4, v5)
    print_vector(v4, "v4")
    print_vector(v5, "v5")
    print_vector(result3, "v4 + v5")
    print("Expected: [4.0, 0.0, 0.0]")

    # Test 4: Shape mismatch (should error!)
    print("\n--- Test 4: Shape mismatch (should fail) ---")
    var v6: List[Float64] = [1.0, 2.0, 3.0]
    var v7: List[Float64] = [4.0, 5.0]  # Different length!

    try:
        var bad_result = vector_add(v6, v7)
        print("ERROR: Should have caught shape mismatch!")
    except e:
        print("✓ Correctly caught shape mismatch!")
        print("  Error message: Shape mismatch detected")

    print("\n" + "="*60)
    print("Vector Addition Tests Complete!")
    print("="*60 + "\n")


# ============================================================================
# MAIN: Run tests
# ============================================================================

fn test_dot_product() raises:
    """Test our dot product implementation."""
    print("\n" + "="*60)
    print("Testing Dot Product")
    print("="*60)

    # Test 1: Basic dot product
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = dot_product(v1, v2)

    print_vector(v1, "v1")
    print_vector(v2, "v2")
    print("v1 · v2 =", result)
    print("Expected: 32.0 (1*4 + 2*5 + 3*6)")

    # Test 2: Orthogonal vectors (dot product = 0)
    print("\n--- Test 2: Orthogonal vectors ---")
    var v3: List[Float64] = [1.0, 0.0]
    var v4: List[Float64] = [0.0, 1.0]
    var result2 = dot_product(v3, v4)
    print_vector(v3, "v3")
    print_vector(v4, "v4")
    print("v3 · v4 =", result2)
    print("Expected: 0.0 (vectors at 90° angle)")

    # Test 3: Dot product with itself (magnitude squared)
    print("\n--- Test 3: Vector with itself ---")
    var v5: List[Float64] = [3.0, 4.0]
    var result3 = dot_product(v5, v5)
    print_vector(v5, "v5")
    print("v5 · v5 =", result3)
    print("Expected: 25.0 (3² + 4² = 9 + 16)")

    # Test 4: Negative numbers
    print("\n--- Test 4: Negative numbers ---")
    var v6: List[Float64] = [1.0, -2.0, 3.0]
    var v7: List[Float64] = [4.0, 5.0, -6.0]
    var result4 = dot_product(v6, v7)
    print_vector(v6, "v6")
    print_vector(v7, "v7")
    print("v6 · v7 =", result4)
    print("Expected: -24.0 (1*4 + (-2)*5 + 3*(-6) = 4 - 10 - 18)")

    # Test 5: Zero vector
    print("\n--- Test 5: Zero vector ---")
    var v8: List[Float64] = [1.0, 2.0, 3.0]
    var zeros: List[Float64] = [0.0, 0.0, 0.0]
    var result5 = dot_product(v8, zeros)
    print_vector(v8, "v8")
    print_vector(zeros, "zeros")
    print("v8 · zeros =", result5)
    print("Expected: 0.0 (any vector · zero vector = 0)")

    print("\n" + "="*60)
    print("Dot Product Tests Complete!")
    print("="*60 + "\n")


fn main() raises:
    print("\n🔥 Block 0: Vector Operations 🔥")
    print("Building ML fundamentals from scratch in Mojo\n")

    # Run vector addition tests
    test_vector_add()

    # Run dot product tests
    test_dot_product()

    print("\n📚 Key Takeaways:")
    print("  1. Vector addition: element-wise operation → returns vector")
    print("  2. Dot product: multiply + sum → returns scalar")
    print("  3. Dot product = 0 means vectors are orthogonal (perpendicular)")
    print("  4. Dot product with itself = magnitude squared")
    print("  5. Dot product is the foundation of neural networks!")
    print("\n💡 Next: We'll optimize dot product with SIMD!")
