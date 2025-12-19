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
