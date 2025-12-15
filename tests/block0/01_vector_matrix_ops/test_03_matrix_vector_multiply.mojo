"""
Tests for Matrix-Vector Multiply.
Run with: pixi run mojo tests/block0/01_vector_matrix_ops/test_03_matrix_vector_multiply.mojo
"""

from testing import assert_equal, assert_raises

# Copy functions for testing
fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    """Compute dot product."""
    if len(a) != len(b):
        raise Error("Shape mismatch")
    var result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


fn matrix_vector_multiply(
    matrix: List[List[Float64]],
    vector: List[Float64]
) raises -> List[Float64]:
    """Multiply matrix by vector."""
    if len(matrix) > 0 and len(matrix[0]) != len(vector):
        raise Error("Shape mismatch: matrix columns must equal vector length")

    var result = List[Float64]()
    for i in range(len(matrix)):
        var row_result = dot_product(matrix[i], vector)
        result.append(row_result)
    return result^


# ============================================================================
# TESTS
# ============================================================================

fn test_basic() raises:
    """Test basic matrix-vector multiply."""
    print("  Testing: Basic (3×2) @ (2×1)...", end=" ")

    var matrix: List[List[Float64]] = [
        [1.0, 2.0],
        [3.0, 4.0],
        [7.0, 8.0]
    ]
    var vector: List[Float64] = [5.0, 6.0]
    var result = matrix_vector_multiply(matrix, vector)

    assert_equal(len(result), 3)
    assert_equal(result[0], 17.0)  # 1*5 + 2*6
    assert_equal(result[1], 39.0)  # 3*5 + 4*6
    assert_equal(result[2], 83.0)  # 7*5 + 8*6
    print("✓")


fn test_single_row() raises:
    """Test with single row (1×n matrix)."""
    print("  Testing: Single row...", end=" ")

    var matrix: List[List[Float64]] = [[2.0, 3.0, 4.0]]
    var vector: List[Float64] = [1.0, 2.0, 3.0]
    var result = matrix_vector_multiply(matrix, vector)

    assert_equal(len(result), 1)
    assert_equal(result[0], 20.0)  # 2*1 + 3*2 + 4*3
    print("✓")


fn test_identity_matrix() raises:
    """Test with identity matrix (should return same vector)."""
    print("  Testing: Identity matrix...", end=" ")

    var identity: List[List[Float64]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ]
    var vector: List[Float64] = [5.0, 7.0, 9.0]
    var result = matrix_vector_multiply(identity, vector)

    assert_equal(len(result), 3)
    assert_equal(result[0], 5.0)
    assert_equal(result[1], 7.0)
    assert_equal(result[2], 9.0)
    print("✓")


fn test_zero_matrix() raises:
    """Test with zero matrix (should return zeros)."""
    print("  Testing: Zero matrix...", end=" ")

    var zeros: List[List[Float64]] = [
        [0.0, 0.0],
        [0.0, 0.0]
    ]
    var vector: List[Float64] = [5.0, 6.0]
    var result = matrix_vector_multiply(zeros, vector)

    assert_equal(len(result), 2)
    assert_equal(result[0], 0.0)
    assert_equal(result[1], 0.0)
    print("✓")


fn test_negative_values() raises:
    """Test with negative numbers (neural network weights)."""
    print("  Testing: Negative values...", end=" ")

    var matrix: List[List[Float64]] = [
        [-1.0, 2.0],
        [3.0, -4.0]
    ]
    var vector: List[Float64] = [5.0, 6.0]
    var result = matrix_vector_multiply(matrix, vector)

    assert_equal(result[0], 7.0)   # -1*5 + 2*6 = -5 + 12
    assert_equal(result[1], -9.0)  # 3*5 + (-4)*6 = 15 - 24
    print("✓")


fn test_shape_mismatch() raises:
    """Test shape mismatch error."""
    print("  Testing: Shape mismatch...", end=" ")

    var matrix: List[List[Float64]] = [[1.0, 2.0, 3.0]]  # 1×3
    var vector: List[Float64] = [5.0, 6.0]  # Length 2 (doesn't match!)

    with assert_raises():
        var result = matrix_vector_multiply(matrix, vector)
    print("✓")


fn test_square_matrix() raises:
    """Test with square matrix (common in ML)."""
    print("  Testing: Square matrix (2×2)...", end=" ")

    var matrix: List[List[Float64]] = [
        [2.0, 3.0],
        [4.0, 5.0]
    ]
    var vector: List[Float64] = [1.0, 2.0]
    var result = matrix_vector_multiply(matrix, vector)

    assert_equal(result[0], 8.0)   # 2*1 + 3*2
    assert_equal(result[1], 14.0)  # 4*1 + 5*2
    print("✓")


# ============================================================================
# TEST RUNNER
# ============================================================================

fn main() raises:
    print("\n" + "="*60)
    print("🧪 Matrix-Vector Multiply Tests")
    print("="*60)

    test_basic()
    test_single_row()
    test_identity_matrix()
    test_zero_matrix()
    test_negative_values()
    test_shape_mismatch()
    test_square_matrix()

    print("\n✅ All tests passed!\n")
