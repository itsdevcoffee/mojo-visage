"""Tests for visage library operations."""

from visage import (
    vector_add, dot_product, matrix_vector_multiply,
    matrix_matrix_multiply, transpose,
    elementwise_multiply, elementwise_divide, scalar_multiply
)


fn test_vector_add() raises:
    """Test vector addition."""
    print("Testing vector_add...")

    # Basic addition
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = vector_add(v1, v2)

    assert_equal(len(result), 3, "Result should have 3 elements")
    assert_equal(result[0], 5.0, "First element should be 5.0")
    assert_equal(result[1], 7.0, "Second element should be 7.0")
    assert_equal(result[2], 9.0, "Third element should be 9.0")

    # Zero vector
    var v3: List[Float64] = [1.0, 2.0]
    var v4: List[Float64] = [0.0, 0.0]
    var result2 = vector_add(v3, v4)
    assert_equal(result2[0], 1.0, "Adding zero should not change value")

    # Negative numbers
    var v5: List[Float64] = [1.0, -2.0]
    var v6: List[Float64] = [-1.0, 2.0]
    var result3 = vector_add(v5, v6)
    assert_equal(result3[0], 0.0, "1 + -1 should be 0")
    assert_equal(result3[1], 0.0, "-2 + 2 should be 0")

    print("  ✓ All vector_add tests passed")


fn test_vector_add_errors() raises:
    """Test vector addition error handling."""
    print("Testing vector_add error handling...")

    var v1: List[Float64] = [1.0, 2.0]
    var v2: List[Float64] = [1.0, 2.0, 3.0]

    var raised = False
    try:
        _ = vector_add(v1, v2)
    except:
        raised = True

    assert_true(raised, "Should raise error for mismatched shapes")
    print("  ✓ Error handling tests passed")


fn test_dot_product() raises:
    """Test dot product."""
    print("Testing dot_product...")

    # Basic dot product
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = dot_product(v1, v2)
    assert_equal(result, 32.0, "Dot product should be 32.0")

    # Orthogonal vectors (perpendicular)
    var v3: List[Float64] = [1.0, 0.0]
    var v4: List[Float64] = [0.0, 1.0]
    var result2 = dot_product(v3, v4)
    assert_equal(result2, 0.0, "Orthogonal vectors have dot product 0")

    # Magnitude squared (vector with itself)
    var v5: List[Float64] = [3.0, 4.0]
    var result3 = dot_product(v5, v5)
    assert_equal(result3, 25.0, "Magnitude squared should be 25")

    # Opposite vectors (negative result)
    var v6: List[Float64] = [1.0, 0.0]
    var v7: List[Float64] = [-1.0, 0.0]
    var result4 = dot_product(v6, v7)
    assert_equal(result4, -1.0, "Opposite vectors should have negative dot product")

    print("  ✓ All dot_product tests passed")


fn test_matrix_vector_multiply() raises:
    """Test matrix-vector multiplication."""
    print("Testing matrix_vector_multiply...")

    # Basic matrix-vector multiply
    var matrix: List[List[Float64]] = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ]
    var vector: List[Float64] = [7.0, 8.0]
    var result = matrix_vector_multiply(matrix, vector)

    assert_equal(len(result), 3, "Result should have 3 elements")
    assert_equal(result[0], 23.0, "Row 1: 1*7 + 2*8 = 23")
    assert_equal(result[1], 53.0, "Row 2: 3*7 + 4*8 = 53")
    assert_equal(result[2], 83.0, "Row 3: 5*7 + 6*8 = 83")

    # Identity-like operation
    var identity: List[List[Float64]] = [[1.0, 0.0], [0.0, 1.0]]
    var v: List[Float64] = [5.0, 3.0]
    var result2 = matrix_vector_multiply(identity, v)
    assert_equal(result2[0], 5.0, "Identity should preserve first element")
    assert_equal(result2[1], 3.0, "Identity should preserve second element")

    print("  ✓ All matrix_vector_multiply tests passed")


fn test_matrix_matrix_multiply() raises:
    """Test matrix-matrix multiplication."""
    print("Testing matrix_matrix_multiply...")

    var A: List[List[Float64]] = [
        [1.0, 2.0],
        [3.0, 4.0]
    ]
    var B: List[List[Float64]] = [
        [5.0, 6.0],
        [7.0, 8.0]
    ]
    var C = matrix_matrix_multiply(A, B)

    # Expected: [[19, 22], [43, 50]]
    assert_equal(C[0][0], 19.0, "C[0,0] should be 19")
    assert_equal(C[0][1], 22.0, "C[0,1] should be 22")
    assert_equal(C[1][0], 43.0, "C[1,0] should be 43")
    assert_equal(C[1][1], 50.0, "C[1,1] should be 50")

    print("  ✓ All matrix_matrix_multiply tests passed")


fn test_transpose() raises:
    """Test matrix transpose."""
    print("Testing transpose...")

    var matrix: List[List[Float64]] = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]
    var T = transpose(matrix)

    # Shape should be 3x2
    assert_equal(len(T), 3, "Transposed should have 3 rows")
    assert_equal(len(T[0]), 2, "Transposed rows should have 2 elements")

    # Check values
    assert_equal(T[0][0], 1.0, "T[0,0] should be 1")
    assert_equal(T[0][1], 4.0, "T[0,1] should be 4")
    assert_equal(T[1][0], 2.0, "T[1,0] should be 2")
    assert_equal(T[1][1], 5.0, "T[1,1] should be 5")
    assert_equal(T[2][0], 3.0, "T[2,0] should be 3")
    assert_equal(T[2][1], 6.0, "T[2,1] should be 6")

    print("  ✓ All transpose tests passed")


fn test_elementwise_operations() raises:
    """Test element-wise operations."""
    print("Testing elementwise operations...")

    var a: List[Float64] = [2.0, 4.0, 6.0]
    var b: List[Float64] = [1.0, 2.0, 3.0]

    # Element-wise multiply
    var prod = elementwise_multiply(a, b)
    assert_equal(prod[0], 2.0, "2*1 should be 2")
    assert_equal(prod[1], 8.0, "4*2 should be 8")
    assert_equal(prod[2], 18.0, "6*3 should be 18")

    # Element-wise divide
    var quot = elementwise_divide(a, b)
    assert_equal(quot[0], 2.0, "2/1 should be 2")
    assert_equal(quot[1], 2.0, "4/2 should be 2")
    assert_equal(quot[2], 2.0, "6/3 should be 2")

    print("  ✓ All elementwise operation tests passed")


fn test_scalar_multiply() raises:
    """Test scalar multiplication."""
    print("Testing scalar_multiply...")

    var vec: List[Float64] = [1.0, 2.0, 3.0]
    var result = scalar_multiply(2.5, vec)

    assert_equal(result[0], 2.5, "2.5*1 should be 2.5")
    assert_equal(result[1], 5.0, "2.5*2 should be 5.0")
    assert_equal(result[2], 7.5, "2.5*3 should be 7.5")

    print("  ✓ All scalar_multiply tests passed")


fn assert_equal(value: Float64, expected: Float64, message: String) raises:
    """Assert that value equals expected."""
    if value != expected:
        raise Error(message)


fn assert_equal(value: Int, expected: Int, message: String) raises:
    """Assert that value equals expected."""
    if value != expected:
        raise Error(message)


fn assert_true(condition: Bool, message: String) raises:
    """Assert that condition is true."""
    if not condition:
        raise Error(message)


fn main() raises:
    """Run all tests."""
    print("\n" + "="*50)
    print("Running Visage ML Library Tests")
    print("="*50 + "\n")

    test_vector_add()
    test_vector_add_errors()
    test_dot_product()
    test_matrix_vector_multiply()
    test_matrix_matrix_multiply()
    test_transpose()
    test_elementwise_operations()
    test_scalar_multiply()

    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50 + "\n")
