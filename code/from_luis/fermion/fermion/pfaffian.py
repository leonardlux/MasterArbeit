import numpy as np
from numba import njit, prange
import time

@njit(parallel=True)
def parlett_reid_pfaffian(A):
    """
    Computes the Pfaffian of an antisymmetric matrix A using the Parlett-Reid algorithm with parallelization.
    
    Parameters:
    - A: np.ndarray, an antisymmetric matrix of even dimension (n x n).
    
    Returns:
    - float: Pfaffian of the matrix with the correct sign.
    """
    n = A.shape[0]

    # Check if matrix is antisymmetric (not enforced within @njit, so do this before calling the function)
    if not np.allclose(A, -A.T):
        raise ValueError("Matrix must be antisymmetric.")

    # Ensure matrix dimensions are even
    if n % 2 != 0:
        raise ValueError("Matrix must have an even dimension to compute the Pfaffian.")

    # Copy the input matrix to avoid modifying the original
    A = A.copy()

    # Initialize the sign of the Pfaffian
    pf_sign = 1.0

    for k in range(0, n - 1, 2):
        if A[k, k + 1] == 0:
            raise ValueError("Zero encountered on the diagonal block, algorithm cannot proceed.")

        # Multiply the Pfaffian sign by the value of the block
        pf_sign *= A[k, k + 1]

        # Parallel Schur complement update
        for i in prange(k + 2, n):
            for j in range(k + 2, n):
                A[i, j] -= (A[i, k] * A[k + 1, j] - A[i, k + 1] * A[k, j]) / A[k, k + 1]

    return pf_sign
  
@njit(parallel=True)
def pfaffian_LTL(A, overwrite_a=False):
    """pfaffian_LTL(A, overwrite_a=False)

    Compute the Pfaffian of a real or complex skew-symmetric
    matrix A (A=-A^T). If overwrite_a=True, the matrix A
    is overwritten in the process. This function uses
    the Parlett-Reid algorithm.
    """
    # Check if matrix is square
    assert A.shape[0] == A.shape[1] > 0
    # Check if it's skew-symmetric
    assert abs((A + A.T).max()) < 1e-14

    n, m = A.shape
    '''
    # type check to fix problems with integer numbers
    dtype = type(A[0, 0])
    if dtype != np.complex128:
        # the slice views work only properly for arrays
        A = np.asarray(A, dtype=float)
    '''

    # Quick return if possible
    if n % 2 == 1:
        return 0

    if not overwrite_a:
        A = A.copy()

    pfaffian_val = 1.0

    for k in range(0, n - 1, 2):
        # First, find the largest entry in A[k+1:,k] and
        # permute it to A[k+1,k]
        kp = k + 1 + np.abs(A[k + 1 :, k]).argmax()

        # Check if we need to pivot
        if kp != k + 1:
            # interchange rows k+1 and kp
            temp = A[k + 1, k:].copy()
            A[k + 1, k:] = A[kp, k:]
            A[kp, k:] = temp

            # Then interchange columns k+1 and kp
            temp = A[k:, k + 1].copy()
            A[k:, k + 1] = A[k:, kp]
            A[k:, kp] = temp

            # every interchange corresponds to a "-" in det(P)
            pfaffian_val *= -1

        # Now form the Gauss vector
        if A[k + 1, k] != 0.0:
            tau = A[k, k + 2 :].copy()
            tau = tau / A[k, k + 1]

            pfaffian_val *= A[k, k + 1]

            if k + 2 < n:
                # Update the matrix block A(k+2:,k+2)
                A[k + 2 :, k + 2 :] = A[k + 2 :, k + 2 :] + np.outer(
                    tau, A[k + 2 :, k + 1]
                )
                A[k + 2 :, k + 2 :] = A[k + 2 :, k + 2 :] - np.outer(
                    A[k + 2 :, k + 1], tau
                )
        else:
            # if we encounter a zero on the super/subdiagonal, the
            # Pfaffian is 0
            return 0.0

    return pfaffian_val

'''
# Example usage
AP = 0.5 * np.array([
    [0, 2, -1, -1],
    [-2, 0, -1, 1],
    [1, 1, 0, 0],
    [1, -1, 0, 0]
])

pf_result = parlett_reid_pfaffian(AP)
print("Pfaffian of AP:", pf_result)

t1 =
pf_result = pfaffian_LTL(AP)
print("Pfaffian of AP:", pf_result)
'''
