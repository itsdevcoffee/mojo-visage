fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    # STEP 1: Shape validation
    if len(a) != len(b):
        raise Error("Shape mismatch: vectors must have same length. "
                   + "Got " + String(len(a)) + " and " + String(len(b)))

    # STEP 2: Initialize accumulator
    var result: Float64 = 0.0

    # STEP 3: Multiply and accumulate (this is the dot product!)
    for i in range(len(a)):
        result += a[i] * b[i]  # Multiply elements, add to sum

    return result

fn matrix_vector_multiply(
    matrix: List[List[Float64]],
    vector: List[Float64]
) raises -> List[Float64]:
    """
    Multiply matrix by vector: each result[i] = matrix[i] · vector.

    This is what neural network layers do!

    Shape: (m × n) @ (n) = (m)
    """
    # Shape validation
    if len(matrix) > 0 and len(matrix[0]) != len(vector):
        raise Error("Shape mismatch: matrix columns must equal vector length. "
                   + "Got " + String(len(matrix[0])) + " cols and " + String(len(vector)) + " elements")

    var result = List[Float64]()

    # For each row, compute dot product with vector
    for i in range(len(matrix)):
        var row_result = dot_product(matrix[i], vector)
        result.append(row_result)

    return result^


fn main() raises:
    """Demo: Matrix-vector multiply (neural network layer!)"""
    print("\n=== Matrix-Vector Multiply Demo ===\n")

    var test_matrix: List[List[Float64]] = [
        [1.0, 2.0],
        [3.0, 4.0],
        [7.0, 8.0]
    ]

    var test_vector: List[Float64] = [5.0, 6.0]

    print("Matrix (3×2):")
    for i in range(len(test_matrix)):
        print("  ", test_matrix[i])

    print("\nVector (2×1):", test_vector)

    var test_result: List[Float64] = matrix_vector_multiply(test_matrix, test_vector)

    print("\nResult (3×1): W @ x =", test_result)
    print("Expected: [17.0, 39.0, 83.0]")
    print("\nBreakdown:")
    print("  Row 1: [1,2] · [5,6] = 1*5 + 2*6 = 17")
    print("  Row 2: [3,4] · [5,6] = 3*5 + 4*6 = 39")
    print("  Row 3: [7,8] · [5,6] = 7*5 + 8*6 = 83")

    print("\n✓ This is what a neural network layer does!")

