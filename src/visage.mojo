"""
Visage ML - Linear Algebra Operations

Core primitives for neural networks and ML.
"""


fn vector_add(a: List[Float64], b: List[Float64]) raises -> List[Float64]:
    """
    Element-wise vector addition: c[i] = a[i] + b[i].

    Args:
        a: First vector
        b: Second vector (must match length of a)

    Returns:
        New vector containing element-wise sum

    Raises:
        Error if vector lengths don't match
    """
    if len(a) != len(b):
        raise Error(
            "Shape mismatch: vectors must have same length. Got "
            + String(len(a)) + " and " + String(len(b))
        )

    var result = List[Float64]()
    for i in range(len(a)):
        result.append(a[i] + b[i])

    return result^


fn dot_product(a: List[Float64], b: List[Float64]) raises -> Float64:
    """
    Compute dot product: a · b = Σ(a[i] * b[i]).

    The fundamental operation in neural networks - used for weighted sums,
    attention mechanisms, similarity metrics, and matrix multiplication.

    Args:
        a: First vector
        b: Second vector (must match length of a)

    Returns:
        Scalar dot product

    Raises:
        Error if vector lengths don't match
    """
    if len(a) != len(b):
        raise Error(
            "Shape mismatch: vectors must have same length. Got "
            + String(len(a)) + " and " + String(len(b))
        )

    var result: Float64 = 0.0
    for i in range(len(a)):
        result += a[i] * b[i]

    return result


fn matrix_vector_multiply(
    matrix: List[List[Float64]],
    vector: List[Float64]
) raises -> List[Float64]:
    """
    Matrix-vector multiplication: result[i] = matrix[i] · vector.

    This is the core operation of a neural network layer:
    output = weights @ input

    Args:
        matrix: m×n matrix (list of row vectors)
        vector: n-dimensional vector

    Returns:
        m-dimensional result vector

    Raises:
        Error if matrix columns ≠ vector length
    """
    if len(matrix) == 0:
        raise Error("Empty matrix")

    if len(matrix[0]) != len(vector):
        raise Error(
            "Shape mismatch: matrix columns must equal vector length. Got "
            + String(len(matrix[0])) + " cols and " + String(len(vector)) + " elements"
        )

    var result = List[Float64]()

    for i in range(len(matrix)):
        var row_result = dot_product(matrix[i], vector)
        result.append(row_result)

    return result^


# ==============================================================================
# Helper Functions
# ==============================================================================

fn get_column(matrix: List[List[Float64]], col_idx: Int) raises -> List[Float64]:
    """
    Extract a column from a matrix.

    Args:
        matrix: m×n matrix
        col_idx: Column index (0 to n-1)

    Returns:
        Column as vector (length m)
    """
    var column = List[Float64]()
    for row_idx in range(len(matrix)):
        column.append(matrix[row_idx][col_idx])
    return column^


# ==============================================================================
# Matrix Operations
# ==============================================================================

fn matrix_matrix_multiply(
    A: List[List[Float64]],
    B: List[List[Float64]]
) raises -> List[List[Float64]]:
    """
    Matrix-matrix multiplication: C = A @ B

    Powers neural network layers and transformations.

    Shape rules:
        A: (m × n)
        B: (n × p)
        C: (m × p)

    Args:
        A: Left matrix (m × n)
        B: Right matrix (n × p)

    Returns:
        Result matrix (m × p)

    Raises:
        Error if A's columns ≠ B's rows
    """
    if len(A) == 0 or len(B) == 0:
        raise Error("Empty matrix")

    var A_cols = len(A[0])
    var B_rows = len(B)

    if A_cols != B_rows:
        raise Error(
            "Shape mismatch: A columns must equal B rows. Got A: (?, "
            + String(A_cols) + ") and B: (" + String(B_rows) + ", ?)"
        )

    var m = len(A)
    var p = len(B[0])
    var result = List[List[Float64]]()

    for i in range(m):
        var row = List[Float64]()
        for j in range(p):
            var B_column = get_column(B, j)
            var value = dot_product(A[i], B_column)
            row.append(value)
        result.append(row^)

    return result^


fn transpose(matrix: List[List[Float64]]) raises -> List[List[Float64]]:
    """
    Transpose a matrix: swap rows and columns.

    Essential for backpropagation and matrix operations.

    Args:
        matrix: m×n matrix

    Returns:
        Transposed n×m matrix
    """
    if len(matrix) == 0:
        raise Error("Empty matrix")

    var rows = len(matrix)
    var cols = len(matrix[0])
    var result = List[List[Float64]]()

    for j in range(cols):
        var new_row = get_column(matrix, j)
        result.append(new_row^)

    return result^


# ==============================================================================
# Element-wise Operations
# ==============================================================================

fn elementwise_multiply(a: List[Float64], b: List[Float64]) raises -> List[Float64]:
    """
    Element-wise multiplication (Hadamard product): c[i] = a[i] * b[i]

    Used in backpropagation and gating mechanisms.

    Args:
        a: First vector
        b: Second vector (must match length)

    Returns:
        Element-wise product
    """
    if len(a) != len(b):
        raise Error(
            "Shape mismatch: vectors must have same length. Got "
            + String(len(a)) + " and " + String(len(b))
        )

    var result = List[Float64]()
    for i in range(len(a)):
        result.append(a[i] * b[i])
    return result^


fn elementwise_divide(a: List[Float64], b: List[Float64]) raises -> List[Float64]:
    """
    Element-wise division: c[i] = a[i] / b[i]

    Args:
        a: Numerator vector
        b: Denominator vector (must match length, no zeros)

    Returns:
        Element-wise quotient
    """
    if len(a) != len(b):
        raise Error(
            "Shape mismatch: vectors must have same length. Got "
            + String(len(a)) + " and " + String(len(b))
        )

    var result = List[Float64]()
    for i in range(len(a)):
        if b[i] == 0.0:
            raise Error("Division by zero at index " + String(i))
        result.append(a[i] / b[i])
    return result^


fn scalar_multiply(scalar: Float64, vector: List[Float64]) -> List[Float64]:
    """
    Multiply vector by scalar: result[i] = scalar * vector[i]

    Used for learning rate scaling and normalization.

    Args:
        scalar: Scalar multiplier
        vector: Input vector

    Returns:
        Scaled vector
    """
    var result = List[Float64]()
    for i in range(len(vector)):
        result.append(scalar * vector[i])
    return result^
