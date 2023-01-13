def append_mat_right(mat1, mat2):

    # takes two matrices of form outlined in flowerbox, must be same size
    # returns one auigmented matrix of mat1 on left and mat2 on right

    # appends a matrix (mat2) to the right of another matrix (mat1) such that we have essentially created an augmented matrix

    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]): # if the dimensions are compatinle
        for idx, row in enumerate(mat2): # we iterate throgh thhe second matrix O(n)
            mat1[idx].extend(row) # and add it to the first via the extend fucntion, O(n)

    return mat1 # we return the augmented mat 1

    # full time O(n**2)

def get_col(matrix_2d, _index):

    return list(row[_index] for row in matrix_2d) # O(n) this simply grabs the column from the specified index. 

    # full time O(n)


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

def make_identity(dim):

    # this codes creates an identity matrix with the specifid dimensions

    l = [] # creates empty list

    for one_idx in range(dim):
        new_row = [0 for i in range(dim)] # makes sublist with all 0s
        new_row[one_idx] = 1 # replaces the diagonal with a 1
        l.append(new_row) # adds to the list we will return
    
    return l # returns the needed list
    # full time: O(n**2)

def mround(matrix, places=2):

    for idx, row in enumerate(matrix):
        for idx2, column in enumerate(row):
            matrix[idx][idx2] = round(column, places)

    return matrix

   # O(n**2)

g = [
[0.0, -0.8660254037844385, -0.4082482904638624, -0.28867513459481053],
[-0.4082482904638631, 0.28867513459481275, -0.8164965809277263, 0.2886751345948111],
[0.8164965809277261, 0.288675134594813, -0.40824829046386313, -0.2886751345948134],
[0.4082482904638631, -0.28867513459481275, 1.8129866073473576e-16, 0.8660254037844398]
]

i = make_identity(len(g))

g = append_mat_right(g, i)

h = rref(g)

for row in h:
    print(row)