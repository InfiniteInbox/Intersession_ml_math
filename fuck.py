import matrices as mp
def basisisisisisisi(A1, B1, C1, B_new, C_new):
    def mat_mult(A, B, C):
        # Check that the matrices are compatible for multiplication
        if len(A[0]) != len(B):
            raise ValueError("Incompatible matrices")
        if len(B[0]) != len(C[0]):
            raise ValueError("Incompatible matrices")

        # Initialize the result matrix
        result = [[0 for _ in range(len(C[0]))] for _ in range(len(A))]

        # Perform matrix multiplication
        for i in range(len(A)):
            for j in range(len(C[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j] * C[k][j]

        # Return the result matrix
        return result
    
    def row_reduce(mat1, mat2):
    # Augment the matrices
        A = [mat1[i] + mat2[i] for i in range(len(mat1))]

    # Perform row reduction
        for i in range(len(A)):
            pivot = A[i][i]
            if pivot == 0:
        # Swap rows to find a non-zero pivot element
                for k in range(i + 1, len(A)):
                    if A[k][i] != 0:
                        A[i], A[k] = A[k], A[i]
                        pivot = A[i][i]
                        break
            for j in range(len(A[0])):
                A[i][j] /= pivot
            for k in range(len(A)):
                if k == i:
                    continue
            factor = A[k][i]
            for j in range(len(A[0])):
                A[k][j] -= factor * A[i][j]

  # Return the row-reduced augmented matrix
        mat1_result = [[A[i][j] for j in range(len(mat1[0]))] for i in range(len(A))]
        mat2_result = [[A[i][j + len(mat1[0])] for j in range(len(mat2[0]))] for i in range(len(A))]
        return mat2_result


    Transform_mat_1 = row_reduce(B_new, B1)
    Transform_mat_2 = row_reduce(C_new, C1)

    Final_result = mat_mult(mp.inverse(Transform_mat_1) , A1, Transform_mat_2)

    return Final_result




