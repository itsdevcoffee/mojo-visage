"""
Block 0 - Section 1: Vector/Matrix Operations
Project 4: Matrix-Matrix Multiply

Goal: The complete operation used in neural network layers!
This is what powers deep learning.
"""

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    """Compute dot product (reused from previous projects)."""
    if len(a) != len(b):
        raise Error("Shape mismatch")
    var result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]
    return result


fn get_column(matrix: List[List[Float64]], col_idx: Int) raises -> List[Float64]:
    """
    Extract a column from a matrix.

    Example:
        Matrix:     Column 0:
        [1  2]      [1]
        [3  4]  →   [3]
        [5  6]      [5]

    Args:
        matrix: 2D list (m × n)
        col_idx: Which column to extract (0 to n-1)

    Returns:
        Column as 1D list (length m)
    """
    var column = List[Float64]()

    # Extract element from each row at column index
    for row_idx in range(len(matrix)):
        column.append(matrix[row_idx][col_idx])

    return column^


# ============================================================================
# MATRIX-MATRIX MULTIPLY
# ============================================================================

fn matrix_matrix_multiply(
    A: List[List[Float64]],
    B: List[List[Float64]]
) raises -> List[List[Float64]]:
    """
    Multiply two matrices: C = A @ B

    This is what neural network layers compute!

    Shape rules:
        A: (m × n)
        B: (n × p)
        C: (m × p)

    How it works:
        C[i,j] = row i of A · column j of B

    Args:
        A: Left matrix (m × n)
        B: Right matrix (n × p)

    Returns:
        Result matrix (m × p)

    Raises:
        Error if A's columns ≠ B's rows (inner dimensions must match)
    """
    # STEP 1: Shape validation
    # A's columns must equal B's rows
    if len(A) == 0 or len(B) == 0:
        raise Error("Empty matrix")

    var A_cols = len(A[0])  # Number of columns in A
    var B_rows = len(B)     # Number of rows in B

    if A_cols != B_rows:
        raise Error("Shape mismatch: A columns must equal B rows. "
                   + "Got A: (?, " + String(A_cols) + ") and B: (" + String(B_rows) + ", ?)")

    # STEP 2: Get dimensions
    var m = len(A)        # Number of rows in A (and C)
    var n = len(A[0])     # Number of columns in A = rows in B
    var p = len(B[0])     # Number of columns in B (and C)

    # Result will be (m × p)

    # STEP 3: Create result matrix (m × p)
    var result = List[List[Float64]]()  # 2D list for matrix

    # STEP 4: For each row in A:
    #   For each column in B:
    #     C[i,j] = row i of A · column j of B

    for i in range(m):  # For each row of A
        var row = List[Float64]()  # Create a row for the result

        for j in range(p):  # For each column of B
            # Get column j from B
            var B_column = get_column(B, j)

            # Compute dot product: A[i] · B[:,j]
            var value = dot_product(A[i], B_column)

            # Store in result row
            row.append(value)

        # Add this row to result matrix
        result.append(row^)

    return result^


# ============================================================================
# MAIN: Test the implementation
# ============================================================================

fn main() raises:
    """Demo: Matrix-matrix multiply."""
    print("\n=== Matrix-Matrix Multiply Demo ===\n")

    # Example from explanation above
    var A: List[List[Float64]] = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ]

    var B: List[List[Float64]] = [
        [7.0,  8.0],
        [9.0, 10.0],
        [11.0, 12.0]
    ]

    print("Matrix A (2×3):")
    for row in A:
        print("  ", row)

    print("\nMatrix B (3×2):")
    for row in B:
        print("  ", row)

    var C = matrix_matrix_multiply(A, B)

    print("\nResult C = A @ B (2×2):")
    for row in C:
        print("  ", row)

    print("\nExpected:")
    print("  [58.0, 64.0]")
    print("  [139.0, 154.0]")

    print("\n✓ This is the foundation of deep learning!")
