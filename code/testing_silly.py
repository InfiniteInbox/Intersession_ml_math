# # # mindim = 3

# # # for idx in reversed(range(mindim)):
# # #     # if matrix[idx][idx] == 1:
# # #         for idx2 in reversed(range(idx)):
# # #             print(f'{idx}, {idx2}')

# # # l = [0,1,2,3,4]
# # # l2 = [0,1,12,3,4,1]
# # # for i in l:
# # #     try:
# # #         print(l.index(i))
# # #     except ValueError:
# # #         continue

# # # # for i in range(len(l)):
# # # #     for j in l[i+1:]:
# # # #         print(f"{i}, {j}")

# # # h = True if (1+1==4) else False
# # # print(h)

# # a = [
# #     [2,3],
# #     [4,3],
# #     [3,4]
# # ]

# # a.insert(1, "bababooey")

# # for row in a:
# #     print(row)

# # a = [2,3,54,4,3,2]

# # del a[2:2+2]

# # g = "bruh"
# # h = "abcs"

# # a.insert(2, g)

# # print(a)
# # a.insert(2, h)

# # print(a)

# '''
# imran iftikar
# 11.20.21

# This file is a compilation file - essentially a "library" of all of the matrix functions I have written
# For information on each individual file, consult "./indiv_funcs/", a directory containing each individual file as i wrote them
# In the future, that will likely not be maintained.

# In general, this "library" is to be used with TWO DIMENSIONAL arrays in the form:

#     array = [
#         [a1, a2, a3, ..., aN],
#         [b1, b2, b3, ..., bN],
#         [c1, c2, c3, ..., cN],
#         ...,
#         [z1, z2, z3, ..., zN],
#     ]

# currently, there is support for:
# *   Multiplication by scalar
# *   Mutliplication of matrices
# *   Addition of matrices
# *   subtraction of matrices
# *   tranpose
# *   echelon
# *   determinants
# *   inverse

# I have made an effort to provide good documentation. Where I felt it was necessary, I included time complexity of code
# Keep in mind that although I've not found many bugs, that does not mean there are not cases under which this code will break

# log.txt separate

# known bugs:    
#     little to no argument validation
#     inverse matrix has no clause for singular matrices

# '''

# ######### Dependencies

# ''' none '''

# ######### Main

# def matrix_by_scalar(matrix1, scalar_quantity):

#     if (isinstance(scalar_quantity, int)) or (isinstance(scalar_quantity, float)): # O(1) checks if the second arg is valid

#         return  list([element * scalar_quantity for element in row] \
#                 for row in matrix1) # O(n**2) Multiplies each element of each row by the scalar and thus has a time complexity of n**2
    
#     else: raise ValueError(f"Argument passed: '{scalar_quantity}'. Error: Expected argument of type 'int' or 'float' ") # error raised

#     # full time O(n**2)

# def add_matrices(mat1, mat2): 

#     if (len(mat1) == (len(mat2)) and (len(mat1[0]) == len(mat2[0]))): # check to see if possible to add the two lists. constant time i think

#         # the following is a rather long list comprehension. it just adds each row and col of the two lists together
#         # basically just two for loops
#         l = list([mat1[row][col]+mat2[row][col] \
#             for col in range(len(mat1[0]))] \
#             for row in range(len(mat1))) # O(n**2)
#         return l
    
#     else:
#         raise ValueError("Args are different size and thus cannot be added")
    
#     # full time O(n**2)

# def subtract_matrices(mat1, mat2): # subtracts second term from first

#     if (len(mat1) == (len(mat2)) and (len(mat1[0]) == len(mat2[0]))): # check to see if possible to add the two lists. constant time i think

#         # the following is a rather long list comprehension. it just adds each row and col of the two lists together
#         # basically just two for loops
#         l = list([mat1[row][col]-mat2[row][col] \
#             for col in range(len(mat1[0]))] \
#             for row in range(len(mat1))) # O(n**2)
#         return l

#     else:
#         raise ValueError("Args are different size and thus cannot be subtraction")
    
#     # full time, O(n**2)

# def get_col(matrix_2d, _index):

#     return list(row[_index] for row in matrix_2d) # O(n) this simply grabs the column from the specified index. 

#     # full time O(n)

# def transpose(matrix):

#     new_array = [get_col(matrix, i) for i in range(len(matrix[0]))] # O(n) and nested O(n), becomes O(n**2). takes a column and makes it a row
#     return new_array

#     # full time O(n**2)

# def row_by_scalar(row, scalar_quantity):

#     # the functionality of this function is used in the echelon and inverse function.

#     if (isinstance(scalar_quantity, int)) or (isinstance(scalar_quantity, float)): # checks if argument 2 is valid
#         return list((element*scalar_quantity) for element in row) # simple list comprehension. linear time, O(n)
    
#     else: raise ValueError(" cannot multiply row by non-integer or non-float value") # raises erorr if arg2 is invalid

#     # full time O(n)

# def subtract_row(row1, row2):

#     # this is used in echelon and inverse

#     if len(row1) == len(row2): # makes sure that args can be subtracted
        
#         return list((row1[i] - row2[i]) for i in range(len(row1))) # list comprehension with linear time, O(n)

#     else:
#         raise ValueError("Rows are different sizes and cannot be subtracted") # raises error is size of rows is different

#     # full time O(n)

# def echelon(matrix): 
    
#     for col_index in range(len(matrix[0])): # Indiv O(n), Overall (n**3) the formula I devised uses columns, so I start with that. 
        
#         col = get_col(matrix, col_index) # O(n) we grab the column using the index from the above for loop

#         '''
        
#         the following bit of code looks for places where there might be zeroes in the diagonal.
#         if there are, and we do not handle for it, we get a dividing by zero error

#         thus, the following code is quite necessary.  
        
#         '''
#         if col_index <= len(matrix): # O(1) we only need to look for zeroes in the first square - that is, if the matrix is longer than tall, it is uncessary to check all columns

#             if all((i == 0) for i in col[col_index:]): #O(n) if the entire column is filled with zeroes, we call continue and the program returns to the initial for loop, and goes to the next column
#                 continue 
            
#             elif col[col_index] == 0: # O(1) if one of the elements on the diagonal is zero - this is where the dividing by zero error occurs so we need to handle this
#                 ''' 
#                 here we iterate through all of the rows below the diagonal. 
#                 if we find a row that doesn't contain a zero in the diagonal column index, 
#                 we will swap them
#                 '''
#                 for i in range(len(col[col_index:])): #O(n) 
#                     if col[col_index:][i] != 0: # O(1)
#                         row_idx = col_index+i # O(1)
#                         break 
                
#                 # the below line of code simple swaps the rows
#                 matrix[col_index], matrix[row_idx] = matrix[row_idx], matrix[col_index] # O(1)


#         '''
#         the following for loop is where the actual formula happens
#         the algorithm works as follows:

#         assume we have an array, 
#         we have the nth column and we want to make all elements of that column below 
#         the nth row into 0

#         say n =0 then we have:

#         c1 = [           desired_c1 = [
#               3                         3
#               4                         4
#               3                         0
#               7                         0
#               2                         0
#                 ]                         ]
        
#         we can achieve this via subtracting a scalar multiple of the row such that we get 0
#         ex. row 2. we can achieve row 1, col 2 equaling 0 by 
#         subtracting row 1 * matrix[row1_idx][col2_idx]/matrix[row1_idx][row_idx]
        
#         '''
#         for row_index in range(len(col)): # O(n) we iterate through row of each colum we have grabbed earlier

#             '''
#             remember, we only want to turn the rows below
#             the diagonal into 0. thus, we check if the row is indeed one we one to turn into 0
#             if it is not, its idx will be less than the column idx
#             if that proves to be true, we will simple pass
#             '''

#             if row_index <= col_index: #O(1)n checks if the row is one we do not want to turn to 0
#                 '''
#                 the following if statement is unnecessary, as I could have explicity called:
#                     matrix[col_index][col_index] when I called denominator later
#                     matrix[col_index] when I call raw_subtractant_row later
#                     however, will keep this code for readability, as I find this easier to understand.
#                 '''
#                 if row_index == col_index: #O(1)
#                     denominator = matrix[row_index][col_index] #O(1)
#                     raw_subtractant_row = matrix[row_index] #O(1)
#                 pass              

#             else:
#                 '''
#                 here we actually do the conversion to 0
#                 this finds teh numerator of the scalar we will multiply the subtractant row by
#                 then we will simply create the final subtractant row
#                 then we simply subtract the two rows, resulting in a 0
#                 we then replace the old row with the new one.                
#                 '''
#                 row_to_sub_from = matrix[row_index] # O(1)
#                 numerator = matrix[row_index][col_index] #O(1)


#                 subtractant = row_by_scalar(raw_subtractant_row, (numerator/denominator)) # O(n)
#                 subbed_row = subtract_row(row_to_sub_from, subtractant) # O(1)

#                 matrix[row_index] = subbed_row # O(1)
    
#     return matrix
#     # Full time: O(n**3)

# def matrix_det(matrix, _istriangle = False):

#     if isinstance(_istriangle, bool): # makes sure _istriangle is a boolean
#         while True: # while loop makes sure 
#             if _istriangle == False: # O(1) if tha matrix is not a triangle. it calls echelon
#                 matrix = echelon(matrix) # O(n**3) sets the matrix to a newly triangled matrix 
#                 _istriangle = True # O(1) sets _istriangle to True, as the matrix is now a triangle
#             elif _istriangle == True: # if the matrix is indeed a triangle, we run the following

#                 if len(matrix) == len(matrix[0]): # checks if matrix is square

#                     det = 1 # determinant to 1
#                     for diag_index in range(len(matrix)): # O(n)
#                         det = det*matrix[diag_index][diag_index] # multiplies all of the diagonal elements to find the determinant
                
#                     return det # returns the determinant, breaks the loops
                
#                 else:
#                     print(f"Error: argument expected to be square")
#     else:
#         raise(f"Argument _istriangle must be a bool") # raises error is _istriangle is not boolean
    
#     # Full time: O(n**3)

# def flip(matrix):

#     # the following program flips a matrix's column with a time complexity of O(n**2)
#     matrix = transpose(matrix) # O(n**2)
#     n = len(matrix) # O(1) easier to do than by calling len(matrix each time)

#     holder = [i for i in range(n)] # O(n) a blank list with all of the spots we will need
#     for i in range(0,n): # O(n)
#         holder[i] = matrix[(n-1-i)] # flips each row by assigning opposite idx to the new holder list

#     return transpose(holder) # O(n**2) re-transposes

#     # Full time O(n**2)

# def reflect(matrix):
#     '''
#     the following code reflects matrix about its secondary diagonal
#     by fliping and then transposing and doing that twice
#     total time: O(n**2)
#     '''
#     for i in range(2): # O(1)
#         matrix = flip(matrix) # O(n**2)
#         matrix = transpose(matrix) # O(n**2)
#     return matrix
#     # full time O(n**2)

# def make_identity(dim):

#     # this codes creates an identity matrix with the specifid dimensions

#     l = [] # creates empty list

#     for one_idx in range(dim):
#         new_row = [0 for i in range(dim)] # makes sublist with all 0s
#         new_row[one_idx] = 1 # replaces the diagonal with a 1
#         l.append(new_row) # adds to the list we will return
    
#     return l # returns the needed list
#     # full time: O(n**2)

# def inverse(matrix): 

#     '''
#     finds the inverse of a matrix, uses very similar code to echelon
#     it takes reduces a matrix to upper triangle and applies the same transforms to an indentity matrix
#     it then reflects both of them, and puts the reflect matrix into upper triangle as well, and also
#     applies the changes to the identity matrix. we reflect again, then we divide each each diagonal
#     and apply that to the identity as well. the original matrix becomes an identity matrix,
#         array = [
#                 [1,0,0,0...],
#                 [0,1,0,0...],
#                 [0,0,1,0...],
#                 [0,0,0,0...1]
#         ]
#     and the identity matrix we initialized becomes the matrices inverse. we return the identity matrix
    
#     '''

#     identity = make_identity(len(matrix)) # O(n**2) creates an identity matrix

#     for i in range(2): # O(1), we run through this twice, because we upper triangle and reflect twice

#         ''' the following code is copypastad from echelon save for a few lines
#         so consult echelon for more detailed documentation'''
        
#         for col_index in range(len(matrix[0])): #O(n) this first for loop handles zeroes that might potentially lead to div by 0 errors
#             col = get_col(matrix, col_index) # O(n)

#             if col_index <= len(matrix): # O(1)

#                 if all((i == 0) for i in col[col_index:]): #O(n)
#                     continue 
                
#                 elif col[col_index] == 0: # O(1)
#                     for i in range(len(col[col_index:])): #O(n)
#                         if col[col_index:][i] != 0: # O(1)
#                             row_idx = col_index+i # O(1)
#                             break 
                    
#                     matrix[col_index], matrix[row_idx] = matrix[row_idx], matrix[col_index] # O(n)


#             for row_index in range(len(col)): # O(n)

#                 if row_index <= col_index: #O(1)
#                     if row_index == col_index: #O(1)
#                         denominator = matrix[row_index][col_index] #O(1)
#                         raw_subtractant_row = matrix[row_index] #O(1)
#                         raw_subtractant_row_identity = identity[row_index] # O(1) this line is one of the main ones that differs from echelon
#                     pass              

#                 else:

#                     numerator = matrix[row_index][col_index] #O(1)

#                     row_to_sub_from = matrix[row_index] # O(1)
#                     subtractant = row_by_scalar(raw_subtractant_row, (numerator/denominator)) # O(n)
#                     subbed_row = subtract_row(row_to_sub_from, subtractant) # O(1)
#                     matrix[row_index] = subbed_row

                    
#                     '''
#                     the following three lines are mainly what differs between this and echelon. it simply takes
#                     the operation we did on the row of the argument "matrix" and does it to the row of the identity
#                     '''
#                     subtractant_identity = row_by_scalar(raw_subtractant_row_identity, (numerator/denominator))
#                     subbed_row1 = subtract_row(identity[row_index], subtractant_identity)
#                     identity[row_index] = subbed_row1  
        
#         # the following reflects the matrices so that we can echelon both again. then it reflects it again so we can get back to the original matrix
#         matrix = reflect(matrix)
#         identity = reflect(identity)

#     # we divide by 1/the diagonal in each row of the matrix 
#     for i in range(len(matrix)):
#         identity[i] = row_by_scalar(identity[i], (1/matrix[i][i]))

#     return identity

#     # full time: O(n**3)

# def multip_1row(row1, row2):

#     '''
#     the functionality of this is used in the mutliply matrices program
#     it takes two rows and multiplies them together    
#     '''

#     if len(row1) == len(row2):
#         return sum(list(row1[element]*row2[element] for element in range(len(row1)))) # O(n) n is length of row, they 
#     else:
#         raise ValueError("The dimensions are incomplete")

#     # Full time O(n)

# def multiply_matrix(matrix1, matrix2):

#     '''this function multiplies two matrices together
#     it takes each row and multiplies it by the column in the next row'''

#     # matrix1 row and column number. len() is O(1), constant time
#     matrix1_rows = len(matrix1) 
#     matrix1_cols = len(matrix1[0]) # just taking the first. I assume that each argument is a completely filled list

#     # matrix2 row and column number. still constant time
#     matrix2_rows = len(matrix2)
#     matrix2_cols = len(matrix2[0])

#     # makes sure that we can indeed mutliply these two matrices - that is, their dimensions must be compatible
#     if (matrix1_rows == matrix2_cols) or (matrix1_cols == matrix2_rows): 

#         # if mat1 rows and mat2 cols are equal we switch them so the following code works
#         if (matrix1_rows == matrix2_cols):
#             matrix1, matrix2 = matrix2, matrix1

#         new_matrix = [] # making a new matrix to hold values

#         # iterates through each row of the first matrix
#         for m1_row in matrix1: # this line alone is O(n)
#             new_matrix_row = [] # O(1) creating a new list for just the row we are doing
#             for _index in range(len(matrix2[0])): # this line is O(n)

#                 '''teh following, as a whole, is O(n) we will mutliply the the two rows,
#                 except that one of these these "rows" should be a column
#                 so we will use get_col() to grab it and pass it as an argument. 
#                 this si '''
#                 new_matrix_row.append(multip_1row( # O(n)
#                     m1_row,
#                     get_col(matrix2, _index) # O(n)
#                 ))
            
#             new_matrix.append(new_matrix_row) # addif the new row to a the full matrix

#         return new_matrix # we will return

#     else: # raises the error 
#         raise ValueError("The dimensions of the arrays passed as \
#         arguments are incompatible and cannot be multiplied")
    
#     # full time O(n**3)

# a = [[0.0, -0.8660254037844385, -0.4082482904638624, -0.28867513459481053],
# [-0.4082482904638631, 0.28867513459481275, -0.8164965809277263, 0.2886751345948111],
# [0.8164965809277261, 0.288675134594813, -0.40824829046386313, -0.2886751345948134],
# [0.4082482904638631, -0.28867513459481275, 1.8129866073473576e-16, 0.8660254037844398]]

# b = inverse(a.copy())

# c = multiply_matrix(a,b)

# for row in c:
#     print(row)

# g = 14 + 2j

# c = 3 * g


import matrices as mp

a = [
    [1,2],
    [3,4]
]

print(mp.get_col(a, 0))