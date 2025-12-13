"""
Tests for Dot Product.
Run with: pixi run mojo tests/block0/01_vector_matrix_ops/test_02_dot_product.mojo
"""

from testing import assert_equal, assert_raises

# Copy the function for testing (later we'll use imports)
fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    """Compute dot product."""
    if len(a) != len(b):
        raise Error("Shape mismatch: vectors must have same length")
    var result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


fn test_basic() raises:
    """Test basic dot product."""
    print("  Testing: Basic dot product...", end=" ")
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = dot_product(v1, v2)
    assert_equal(result, 32.0)
    print("✓")


fn test_orthogonal() raises:
    """Test orthogonal vectors."""
    print("  Testing: Orthogonal vectors...", end=" ")
    var v1: List[Float64] = [1.0, 0.0]
    var v2: List[Float64] = [0.0, 1.0]
    var result = dot_product(v1, v2)
    assert_equal(result, 0.0)
    print("✓")


fn test_magnitude_squared() raises:
    """Test magnitude squared."""
    print("  Testing: Magnitude squared...", end=" ")
    var v: List[Float64] = [3.0, 4.0]
    var result = dot_product(v, v)
    assert_equal(result, 25.0)
    print("✓")


fn test_zero_vector() raises:
    """Test with zero vector."""
    print("  Testing: Zero vector...", end=" ")
    var v: List[Float64] = [1.0, 2.0, 3.0]
    var zeros: List[Float64] = [0.0, 0.0, 0.0]
    var result = dot_product(v, zeros)
    assert_equal(result, 0.0)
    print("✓")


fn main() raises:
    print("\n" + "="*60)
    print("🧪 Dot Product Tests")
    print("="*60)
    test_basic()
    test_orthogonal()
    test_magnitude_squared()
    test_zero_vector()
    print("\n✅ All tests passed!\n")
