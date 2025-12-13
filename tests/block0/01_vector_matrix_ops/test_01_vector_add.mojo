"""
Tests for Vector Addition.
Run with: pixi run mojo tests/block0/01_vector_matrix_ops/test_01_vector_add.mojo
"""

from testing import assert_equal, assert_raises

# Copy the function for testing (later we'll use imports)
fn vector_add(a: List[Float64], b: List[Float64]) raises -> List[Float64]:
    """Add two vectors element-wise."""
    if len(a) != len(b):
        raise Error("Shape mismatch: vectors must have same length")
    var result = List[Float64]()
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result^


fn test_basic() raises:
    """Test basic vector addition."""
    print("  Testing: Basic addition...", end=" ")
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = vector_add(v1, v2)
    assert_equal(result[0], 5.0)
    assert_equal(result[1], 7.0)
    assert_equal(result[2], 9.0)
    print("✓")


fn test_identity() raises:
    """Test adding zeros."""
    print("  Testing: Identity (zeros)...", end=" ")
    var v: List[Float64] = [1.0, 2.0, 3.0]
    var zeros: List[Float64] = [0.0, 0.0, 0.0]
    var result = vector_add(v, zeros)
    assert_equal(result[0], 1.0)
    assert_equal(result[1], 2.0)
    assert_equal(result[2], 3.0)
    print("✓")


fn test_shape_mismatch() raises:
    """Test shape mismatch error."""
    print("  Testing: Shape mismatch...", end=" ")
    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0]
    with assert_raises():
        var result = vector_add(v1, v2)
    print("✓")


fn main() raises:
    print("\n" + "="*60)
    print("🧪 Vector Addition Tests")
    print("="*60)
    test_basic()
    test_identity()
    test_shape_mismatch()
    print("\n✅ All tests passed!\n")
