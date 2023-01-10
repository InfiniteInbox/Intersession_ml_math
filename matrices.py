'''
imran iftikar
11.20.21

This file is a compilation file - essentially a "library" of all of the matrix functions I have written
For information on each individual file, consult "./indiv_funcs/", a directory containing each individual file as i wrote them
In the future, that will likely not be maintained.

In general, this "library" is to be used with TWO DIMENSIONAL arrays in the form:

    array = [
        [a1, a2, a3, ..., aN],
        [b1, b2, b3, ..., bN],
        [c1, c2, c3, ..., cN],
        ...,
        [z1, z2, z3, ..., zN],
    ]

currently, there is support for:
*   Multiplication by scalar
*   Mutliplication of matrices
*   Addition of matrices
*   subtraction of matrices
*   tranpose
*   echelon
*   determinants
*   inverse

I have made an effort to provide good documentation. Where I felt it was necessary, I included time complexity of code
Keep in mind that although I've not found many bugs, that does not mean there are not cases under which this code will break

log.txt separate

known bugs:    
    little to no argument validation
    inverse matrix has no clause for singular matrices

'''

######### Dependencies

''' none '''

######### Main

def matrix_by_scalar(matrix1, scalar_quantity):

    if (isinstance(scalar_quantity, int)) or (isinstance(scalar_quantity, float)): # O(1) checks if the second arg is valid

        return  list([element * scalar_quantity for element in row] \
                for row in matrix1) # O(n**2) Multiplies each element of each row by the scalar and thus has a time complexity of n**2
    
    else: raise ValueError(f"Argument passed: '{scalar_quantity}'. Error: Expected argument of type 'int' or 'float' ") # error raised

    # full time O(n**2)

def add_matrices(mat1, mat2): 

    if (len(mat1) == (len(mat2)) and (len(mat1[0]) == len(mat2[0]))): # check to see if possible to add the two lists. constant time i think

        # the following is a rather long list comprehension. it just adds each row and col of the two lists together
        # basically just two for loops
        l = list([mat1[row][col]+mat2[row][col] \
            for col in range(len(mat1[0]))] \
            for row in range(len(mat1))) # O(n**2)
        return l
    
    else:
        raise ValueError("Args are different size and thus cannot be added")
    
    # full time O(n**2)

def subtract_matrices(mat1, mat2): # subtracts second term from first

    if (len(mat1) == (len(mat2)) and (len(mat1[0]) == len(mat2[0]))): # check to see if possible to add the two lists. constant time i think

        # the following is a rather long list comprehension. it just adds each row and col of the two lists together
        # basically just two for loops
        l = list([mat1[row][col]-mat2[row][col] \
            for col in range(len(mat1[0]))] \
            for row in range(len(mat1))) # O(n**2)
        return l

    else:
        raise ValueError("Args are different size and thus cannot be subtraction")
    
    # full time, O(n**2)

def get_col(matrix_2d, _index):

    return list(row[_index] for row in matrix_2d) # O(n) this simply grabs the column from the specified index. 

    # full time O(n)

def transpose(matrix):

    new_array = [get_col(matrix, i) for i in range(len(matrix[0]))] # O(n) and nested O(n), becomes O(n**2). takes a column and makes it a row
    return new_array

    # full time O(n**2)

def row_by_scalar(row, scalar_quantity):

    # the functionality of this function is used in the echelon and inverse function.

    if (isinstance(scalar_quantity, int)) or (isinstance(scalar_quantity, float)): # checks if argument 2 is valid
        return list((element*scalar_quantity) for element in row) # simple list comprehension. linear time, O(n)
    
    else: raise ValueError(" cannot multiply row by non-integer or non-float value") # raises erorr if arg2 is invalid

    # full time O(n)

def subtract_row(row1, row2):

    # this is used in echelon and inverse

    if len(row1) == len(row2): # makes sure that args can be subtracted
        
        return list((row1[i] - row2[i]) for i in range(len(row1))) # list comprehension with linear time, O(n)

    else:
        raise ValueError("Rows are different sizes and cannot be subtracted") # raises error is size of rows is different

    # full time O(n)

def echelon(matrix): 
    
    for col_index in range(len(matrix[0])): # Indiv O(n), Overall (n**3) the formula I devised uses columns, so I start with that. 
        
        col = get_col(matrix, col_index) # O(n) we grab the column using the index from the above for loop

        '''
        
        the following bit of code looks for places where there might be zeroes in the diagonal.
        if there are, and we do not handle for it, we get a dividing by zero error

        thus, the following code is quite necessary.  
        
        '''
        if col_index <= len(matrix): # O(1) we only need to look for zeroes in the first square - that is, if the matrix is longer than tall, it is uncessary to check all columns

            if all((i == 0) for i in col[col_index:]): #O(n) if the entire column is filled with zeroes, we call continue and the program returns to the initial for loop, and goes to the next column
                continue 
            
            elif col[col_index] == 0: # O(1) if one of the elements on the diagonal is zero - this is where the dividing by zero error occurs so we need to handle this
                ''' 
                here we iterate through all of the rows below the diagonal. 
                if we find a row that doesn't contain a zero in the diagonal column index, 
                we will swap them
                '''
                for i in range(len(col[col_index:])): #O(n) 
                    if col[col_index:][i] != 0: # O(1)
                        row_idx = col_index+i # O(1)
                        break 
                
                # the below line of code simple swaps the rows
                matrix[col_index], matrix[row_idx] = matrix[row_idx], matrix[col_index] # O(1)


        '''
        the following for loop is where the actual formula happens
        the algorithm works as follows:

        assume we have an array, 
        we have the nth column and we want to make all elements of that column below 
        the nth row into 0

        say n =0 then we have:

        c1 = [           desired_c1 = [
              3                         3
              4                         4
              3                         0
              7                         0
              2                         0
                ]                         ]
        
        we can achieve this via subtracting a scalar multiple of the row such that we get 0
        ex. row 2. we can achieve row 1, col 2 equaling 0 by 
        subtracting row 1 * matrix[row1_idx][col2_idx]/matrix[row1_idx][row_idx]
        
        '''
        for row_index in range(len(col)): # O(n) we iterate through row of each colum we have grabbed earlier

            '''
            remember, we only want to turn the rows below
            the diagonal into 0. thus, we check if the row is indeed one we one to turn into 0
            if it is not, its idx will be less than the column idx
            if that proves to be true, we will simple pass
            '''

            if row_index <= col_index: #O(1)n checks if the row is one we do not want to turn to 0
                '''
                the following if statement is unnecessary, as I could have explicity called:
                    matrix[col_index][col_index] when I called denominator later
                    matrix[col_index] when I call raw_subtractant_row later
                    however, will keep this code for readability, as I find this easier to understand.
                '''
                if row_index == col_index: #O(1)
                    denominator = matrix[row_index][col_index] #O(1)
                    raw_subtractant_row = matrix[row_index] #O(1)
                pass              

            else:
                '''
                here we actually do the conversion to 0
                this finds teh numerator of the scalar we will multiply the subtractant row by
                then we will simply create the final subtractant row
                then we simply subtract the two rows, resulting in a 0
                we then replace the old row with the new one.                
                '''
                row_to_sub_from = matrix[row_index] # O(1)
                numerator = matrix[row_index][col_index] #O(1)


                subtractant = row_by_scalar(raw_subtractant_row, (numerator/denominator)) # O(n)
                subbed_row = subtract_row(row_to_sub_from, subtractant) # O(1)

                matrix[row_index] = subbed_row # O(1)
    
    return matrix
    
    # Full time: O(n**3)

def ref(matrix):

    # takes matrix of form outlined in flowerbox
    # returns matrix of form outlined in flowerbox, but in ref form

    # builds off of the echelon function, which merely produces an upper triangle
    # ref turns that into an upper triangle such that the leading nonzero element of each row is 1
    
    matrix = echelon(matrix) # we make sure that the input is in upper triangle by applying echelon O(n**3)
    # (this could be put in an if statement to potentially improve time)

    mindim = min(len(matrix), len(matrix[0])) - 1 # here we find the minimum dimension of the matrix

    for idx, row in enumerate(matrix): # we iterate through each row of the matrix and get its index as well O(n)
        # enumerate is O(n) 
        # for loop itself in O(n), so 2O(n) total or just O(n)

        if idx > mindim: # if matrix is longer than wider, we want to finish without iterating through the lower levels
            return matrix # we just return if that is the case, becuase this will be in ref form
        if matrix[idx][idx] !=0: # if the leading element of each row is not 0
            matrix[idx] = row_by_scalar(row, (1/row[idx])) # we will multiply the entire row by the inverse of the leading element O(n)

    return matrix # returns the matrix

    # Full time: O(n**3) with echelon at start, O(n**2 otherwise)

def rref(matrix):

    # takes matrix of form outlined in flowerbox
    # returns matrix of form outlined in flowerbox, but in rref form

    # builds off of the ref function, which produces an upper triangle such that the leading elements of each row is 1
    # rref returns a matrix such that all diagonals are 1 (which is true in ref) but also as many other elements as possible are 0

    matrix = ref(matrix) # we apply ref to the matrix to make sure it is in the appropriate form
    mindim = min(len(matrix), len(matrix[0])) # the minimum dimension of the matrix

    for idx in reversed(range(mindim)): 
        # for above: we wantt to work up from the bottom of the matrix, and will only use values that could have a pivot
        '''in other words, if 
        a = [
        [1,3,8],
        [12,3,4],
        [16,7,1],
        [2,1,6],
        [12,4,5],
        [7,2,5]
        ]
        then only the top 3 rows will have good pivots

        O(n)
        '''

        if matrix[idx][idx] == 1.0: # we know that where pivots should be will only be 0s or 1s becuase of calling ref
                                    # if it is not a one and is a 0, then we will not evaluate and will just skip in
            for idx2 in reversed(range(idx)): # we then iterate through all above rows O(n)
                
                # here we find what we shoudl multiply matrix[idx] by to subtract it from matrix[idx2]
                # and then we simply run through that subtraction
                scalar = matrix[idx2][idx]  # O(1)
                subtractant = row_by_scalar(matrix[idx], scalar) #O(n)
                row_to_sub_from = matrix[idx2] #(1)
                subbed_row = subtract_row(row_to_sub_from, subtractant) # O(n)
    
                matrix[idx2] = subbed_row # O(1)
    
    return matrix

    # Full time O(n**3)

def rank(matrix):

    # takes a matrix
    # returns integer denoting the rank of the matric

    return len(identify_pivots(matrix)) # we find all pivot columns and count how many their are; this is rank by definition

def identify_pivots(matrix):

    # takes matrix of form outlined in flowerbox
    # returns a list of all indexes in which there is a pivot column upon row reducing the input matrix

    # identifies the pivot columns of the matrix by row reducing

    matrix = ref(matrix) # we run ref to row reduce it. we do not care about full row reduction because that it is trivial in this case

    pivot_col_idx_list = [] # we initialize a blank list will contain indexes of pivot cols

    for row in matrix: # we iterate through each row
        try: # we add a try in case there are no ones in the row
            first_one = row.index(1) # we find the first appearance of a 1 within the list
            for idx2 in range(first_one): # we then check if all prior elelements are 0
                if row[idx2] != 0: # if one is not a 0, we contiknue the loops
                    continue

            # however, if all elements prior to the first 1 are 0, we will append the index of that 1 to the pivcollist
            pivot_col_idx_list.append(first_one)

        except: # if there are no ones, it can be infered that there is no pivot column in that row, so we continue
            continue 

    return pivot_col_idx_list    

    # O(n**2) time

def matrix_det(matrix, _istriangle = False):

    if isinstance(_istriangle, bool): # makes sure _istriangle is a boolean
        while True: # while loop makes sure 
            if _istriangle == False: # O(1) if tha matrix is not a triangle. it calls echelon
                matrix = echelon(matrix) # O(n**3) sets the matrix to a newly triangled matrix 
                _istriangle = True # O(1) sets _istriangle to True, as the matrix is now a triangle
            elif _istriangle == True: # if the matrix is indeed a triangle, we run the following

                if len(matrix) == len(matrix[0]): # checks if matrix is square

                    det = 1 # determinant to 1
                    for diag_index in range(len(matrix)): # O(n)
                        det = det*matrix[diag_index][diag_index] # multiplies all of the diagonal elements to find the determinant
                
                    return det # returns the determinant, breaks the loops
                
                else:
                    print(f"Error: argument expected to be square")
    else:
        raise(f"Argument _istriangle must be a bool") # raises error is _istriangle is not boolean
    
    # Full time: O(n**3)

def flip(matrix):

    # the following program flips a matrix's column with a time complexity of O(n**2)
    matrix = transpose(matrix) # O(n**2)
    n = len(matrix) # O(1) easier to do than by calling len(matrix each time)

    holder = [i for i in range(n)] # O(n) a blank list with all of the spots we will need
    for i in range(0,n): # O(n)
        holder[i] = matrix[(n-1-i)] # flips each row by assigning opposite idx to the new holder list

    return transpose(holder) # O(n**2) re-transposes

    # Full time O(n**2)

def reflect(matrix):
    '''
    the following code reflects matrix about its secondary diagonal
    by fliping and then transposing and doing that twice
    total time: O(n**2)
    '''
    for i in range(2): # O(1)
        matrix = flip(matrix) # O(n**2)
        matrix = transpose(matrix) # O(n**2)
    return matrix
    # full time O(n**2)

def make_identity(dim):

    # this codes creates an identity matrix with the specifid dimensions

    l = [] # creates empty list

    for one_idx in range(dim):
        new_row = [0 for i in range(dim)] # makes sublist with all 0s
        new_row[one_idx] = 1 # replaces the diagonal with a 1
        l.append(new_row) # adds to the list we will return
    
    return l # returns the needed list
    # full time: O(n**2)

def inverse(matrix): 

    # takes a matrix of form outlined in flowerbox
    # returns a matrix of form outlined in flowerbox, that is the inverse of the original martix

    '''
    finds the inverse of a matrix, uses very similar code to echelon
    it takes reduces a matrix to upper triangle and applies the same transforms to an indentity matrix
    it then reflects both of them, and puts the reflect matrix into upper triangle as well, and also
    applies the changes to the identity matrix. we reflect again, then we divide each each diagonal
    and apply that to the identity as well. the original matrix becomes an identity matrix,
        array = [
                [1,0,0,0...],
                [0,1,0,0...],
                [0,0,1,0...],
                [0,0,0,0...1]
        ]
    and the identity matrix we initialized becomes the matrices inverse. we return the identity matrix
    
    '''

    if len(identify_pivots(matrix)) != len(matrix):
        return False

    identity = make_identity(len(matrix)) # O(n**2) creates an identity matrix

    for i in range(2): # O(1), we run through this twice, because we upper triangle and reflect twice

        ''' the following code is copypastad from echelon save for a few lines
        so consult echelon for more detailed documentation'''
        
        for col_index in range(len(matrix[0])): #O(n) this first for loop handles zeroes that might potentially lead to div by 0 errors
            col = get_col(matrix, col_index) # O(n)

            if col_index <= len(matrix): # O(1)

                if all((i == 0) for i in col[col_index:]): #O(n)
                    continue 
                
                elif col[col_index] == 0: # O(1)
                    for i in range(len(col[col_index:])): #O(n)
                        if col[col_index:][i] != 0: # O(1)
                            row_idx = col_index+i # O(1)
                            break 
                    
                    matrix[col_index], matrix[row_idx] = matrix[row_idx], matrix[col_index] # O(n)


            for row_index in range(len(col)): # O(n)

                if row_index <= col_index: #O(1)
                    if row_index == col_index: #O(1)
                        denominator = matrix[row_index][col_index] #O(1)
                        raw_subtractant_row = matrix[row_index] #O(1)
                        raw_subtractant_row_identity = identity[row_index] # O(1) this line is one of the main ones that differs from echelon
                    pass              

                else:

                    numerator = matrix[row_index][col_index] #O(1)

                    row_to_sub_from = matrix[row_index] # O(1)
                    subtractant = row_by_scalar(raw_subtractant_row, (numerator/denominator)) # O(n)
                    subbed_row = subtract_row(row_to_sub_from, subtractant) # O(1)
                    matrix[row_index] = subbed_row

                    
                    '''
                    the following three lines are mainly what differs between this and echelon. it simply takes
                    the operation we did on the row of the argument "matrix" and does it to the row of the identity
                    '''
                    subtractant_identity = row_by_scalar(raw_subtractant_row_identity, (numerator/denominator))
                    subbed_row1 = subtract_row(identity[row_index], subtractant_identity)
                    identity[row_index] = subbed_row1  
        
        # the following reflects the matrices so that we can echelon both again. then it reflects it again so we can get back to the original matrix
        matrix = reflect(matrix)
        identity = reflect(identity)

    # we divide by 1/the diagonal in each row of the matrix 
    for i in range(len(matrix)):
        identity[i] = row_by_scalar(identity[i], (1/matrix[i][i]))
        
    return identity

    # full time: O(n**3)

def append_mat_right(mat1, mat2):

    # takes two matrices of form outlined in flowerbox, must be same size
    # returns one auigmented matrix of mat1 on left and mat2 on right

    # appends a matrix (mat2) to the right of another matrix (mat1) such that we have essentially created an augmented matrix

    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]): # if the dimensions are compatinle
        for idx, row in enumerate(mat2): # we iterate throgh thhe second matrix O(n)
            mat1[idx].extend(row) # and add it to the first via the extend fucntion, O(n)

    return mat1 # we return the augmented mat 1

    # full time O(n**2)

def find_transition(og_base, new_base, columned=False):

    '''
    TAKES:
        original basis of subspace, og_base
        new basis of subspace, new_base

        it is important to pay mind to the shape of these basis vectors. 
        eg, input [[1,3,4], [1,4,7]] where each sublist is a basis vector will be treated as 3 col vectors, 
        [1,1], [3,4], and [4,7]. we added the columned flag to help with this
    '''

    if columned==False:
        og_base, new_base = transpose(og_base), transpose(new_base)

    # we find a transition matrix that changes the coordinates expressed from one base to another base

    # going from base 1 to base 2
    if len(og_base) == len(new_base) and len(og_base[0]) == len(new_base[0]):

        dim_to_take = len(og_base[0]) # we find what the dimensions of the transmat will be
        total = append_mat_right(new_base, og_base) # we create an augmented matrix with new basis on left and old basis on right
        reduced = rref(total) # row reduce, O(n**3)

        transition_matrix = list() # init our blank transition matrix

        for row in reduced: # we go through the reduced augmat to find what should be added to the transition matrix
            transition_matrix.append(row[-dim_to_take:])

        return transition_matrix
    
    # FUll time O(n**3)

def change_transformation_basis(ogtransfmat, ogb1, ogb2, tildab1, tildab2, columned=False):
    
    '''
    TAKES:
        the transformation matrix/linmapping, ogtransfmat, of standard form outlined in flowerbox
        the original basis of domain, ogb1
        the original basis of codomain, ogb2
        the new basis of domain, tildab1
        the new basis of codomain, tildab2
        AS WITH find_transition, the base inputs can be finicky, the columned flag should help. check that for more indepth docs

    '''

    if columned==False: # we make sure the basis inputs are the correct form ie they are column vectors
        ogb1, ogb2, tildab1, tildab2 = transpose(ogb1),transpose(ogb2),transpose(tildab1),transpose(tildab2) 

    transition1 = find_transition(tildab1, ogb1) # we find transitino matrix from the new domain basis to the old domain basis, O(n**3)
    transition2 = inverse(find_transition(tildab2, ogb2)) # we find inverse of transition matrix from new codomain basis to old codomain basis, O(n**3)

    return multiply_matrix(multiply_matrix(transition2, ogtransfmat), transition1) # we multiply the matrices according to formula, O(n**3)

    # full time O(n**3)

def multip_1row(row1, row2):

    '''
    the functionality of this is used in the mutliply matrices program
    it takes two rows and multiplies them together    
    '''

    if len(row1) == len(row2):
        return sum(list(row1[element]*row2[element] for element in range(len(row1)))) # O(n) n is length of row, they 
    else:
        raise ValueError("The dimensions are incomplete")

    # Full time O(n)

def multiply_matrix(matrix1, matrix2):

    # the first one goes second

    '''this function multiplies two matrices together
    it takes each row and multiplies it by the column in the next row'''

    # matrix1 row and column number. len() is O(1), constant time
    matrix2_rows = len(matrix2) 
    matrix2_cols = len(matrix2[0]) # just taking the first. I assume that each argument is a completely filled list

    # matrix2 row and column number. still constant time
    matrix1_rows = len(matrix1)
    matrix1_cols = len(matrix1[0])

    # makes sure that we can indeed mutliply these two matrices - that is, their dimensions must be compatible
    if (matrix2_rows == matrix1_cols) or (matrix2_cols == matrix1_rows): 

        # if mat1 rows and mat2 cols are equal we switch them so the following code works
        if (matrix2_rows == matrix1_cols):
            matrix2, matrix1 = matrix1, matrix2

        new_matrix = [] # making a new matrix to hold values

        # iterates through each row of the first matrix
        for m2_row in matrix2: # this line alone is O(n)
            new_matrix_row = [] # O(1) creating a new list for just the row we are doing
            for _index in range(len(matrix1[0])): # this line is O(n)

                '''teh following, as a whole, is O(n) we will mutliply the the two rows,
                except that one of these these "rows" should be a column
                so we will use get_col() to grab it and pass it as an argument. 
                this si '''
                new_matrix_row.append(multip_1row( # O(n)
                    m2_row,
                    get_col(matrix1, _index) # O(n)
                ))
            
            new_matrix.append(new_matrix_row) # addif the new row to a the full matrix

        return new_matrix # we will return

    else: # raises the error 
        raise ValueError("The dimensions of the arrays passed as \
        arguments are incompatible and cannot be multiplied")
    
    # full time O(n**3)

def mround(matrix, places=2):

    for idx, row in enumerate(matrix):
        for idx2, column in enumerate(row):
            matrix[idx][idx2] = round(column, places)

    return matrix

   # O(n**2)

def solve_homogeneous(coef_matrix):
    # takes matrix of form outlined in flowerbox
    # returns a matrix using the minus 1 trick to solve homogeneous linear systems

    coef_matrix = rref(coef_matrix) # takes the rref of the coefficient matrix (system of linear eq) O(n**3)
    pivotcols = identify_pivots(coef_matrix) # finds the pivot columns of the rref matrix O(n**2)
    notpiv_cols = list() # since this function is called because of different size dimension matricies, there are going to be necessary added piv-columns
    for i in range(len(coef_matrix[0])): # iterates through the columns O(n)
        if i not in pivotcols: # Iterates through the list of pivot columns
            notpiv_cols.append(i) # appends the non pivot column to the non-pivot col list
    
    if notpiv_cols == []: # if the non-pivot columns are empty (in other words had total rank)
        return [0 for i in range(len(coef_matrix[0]))]

    for idx in notpiv_cols: #Iterate through the non pivot columns
        coef_matrix.insert(idx, [0 if i != idx else -1 for i in range(len(coef_matrix[0]))]) # by the minus 1 trick, we insert a row with minus 1 in the nth index to preserve diagonality
    coef_matrix = transpose(coef_matrix) # we transpose the matrix
    final = [coef_matrix[idx] for idx in notpiv_cols] # we return the solution as a square matrix  
    return final

    # O(n**3)

a = [
    [5,43,1],
    [7,4,32],
    [21,30,89]
]

b = inverse(a)

print(b)