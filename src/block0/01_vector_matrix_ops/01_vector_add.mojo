"""
Block 0 - Section 1: Vector/Matrix Operations
Project 1: Vector Addition

Goal: Understand element-wise operations and shape validation.
"""

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


fn main() raises:
    """Demo: Vector addition."""
    print("\n=== Vector Addition Demo ===\n")

    var v1: List[Float64] = [1.0, 2.0, 3.0]
    var v2: List[Float64] = [4.0, 5.0, 6.0]
    var result = vector_add(v1, v2)

    print("v1: [1.0, 2.0, 3.0]")
    print("v2: [4.0, 5.0, 6.0]")
    print("v1 + v2 =", result)
    print("\nExpected: [5.0, 7.0, 9.0]")
    print("\n✓ Vector addition complete!")
