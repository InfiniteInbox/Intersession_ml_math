'''
yj iai

INPUT AND DATA TYPES:
    There are three main data types used in the following code. 

    The first is a "vector", which we have represented as follows:

        these are simple python lists, ie [x1, ..., xn] is a vector existing in R^n where x1,...,xn are the elements of the list
        We choose to represent them as row vectors due to ease of use. 

        However, sometimes, we will need to explicitly treat them as column vectors, in which case, they look as follows:

        [
        [x1],
        ...,
        [xn]
        ]

        This is because, on occasion, they must be used with our matrix data type, and thus must follow that form. Other times,
        you will see a vector, v = [x1, ..., xn], represented as [v], or as a list with the vector v within it - such specifies it 
        as a 1*n row vector, such that it too can be used with the matrix data type. 

        We try to keep it concrete in the code with our documentation, but be mindful. In general, treat vectors as simple lists with elements
    
    The second data type we use is matrices:

        we use TWO DIMENSIONAL arrays in the form:

            array = [
                [a1, a2, a3, ..., aN],
                [b1, b2, b3, ..., bN],
                [c1, c2, c3, ..., cN],
                ...,
                [z1, z2, z3, ..., zN],
            ]

            which represents

            a1 a2 a3 ... aN
            b1 b2 b3 ... bN
            c1 c2 c3 ... cN
            ...
            z1 z2 z3 ... zN
        
        In other words, we treat each sublist as a ROW VECTOR of the matrix, NOT AS A COLUMN VECTOR!!!!!
        this is extremely important to keep in mind when dealing with certain parts of this program, ie basis
        because many operations will treat basis vectors as COLUMN VECTORS

        we have included a tag in many of our functions, columned, which defaults to false. columned==False assumes
        that the inputs are row vectors and must be treated as column vectors, and thus will transpose them


'''

#################### DEPENDENCIES ####################

import matrices as mp
import trig

#################### GLOBAL VARS ####################

# none

#################### CLASSES ####################

class sub_space:

    '''
    we have created a glass that can somewhat represent a vector subspace
    it works by taking a generating set that will effectively define the subspace
        (note that this is far from perfect and can't truly represent the concept of a vector subspace)
    in the case of being given basis vectors of R^n, it is FACT than the span of those vectors will ALWAYS be a vector space    '''


    def __init__(self, gset, columned = False):

        '''
        TAKES:
            a generating set of vectors. we assume that they are NOT columned, but WILL NOT CHANGE THEM
            the columned flag, automatically set to False, that is, we assume input vectors are row vectors that should be col vecs
                ie gset = [[1,2,3,4], [1,3,9,1]] then if using with other parts of program, they will read vectors [1,1], [2,3], etc 
                rather than [1,2,3,4] and [1,3,9,1]. 
            We do not change the input in self.gset, but will do so when we determine the basis in self.basis      
            '''

        self.columned = columned # columned flag, this is bool
        self.gset = gset # raw generating set, not modified at all
        self.basis = self.create_basis() # we run the self.basis method, which will determine a basis from the given generating set
        self.isorthonormal = self.isorthonormal()
        self.proj_mat = self.find_projection_mat()


    def create_basis(self): 

        # we find the basis of generatign set

        tempgset = self.gset # we create a temporary gset to not modify the original

        if self.columned == False: # if the input generating vectors are row vecs
            tempgset = mp.transpose(tempgset) # we will make then col vecs

        holder = mp.identify_pivots(tempgset) # now we identify pivots of the generating set, O(n**2)

        return [self.gset[i] for i in holder] # then we will return all of the vectors
    
    def isorthonormal(self): 

        if self.columned == False: # if the input basis vecs are row vecs, a
            basis = mp.transpose(self.basis) # we will make them col vecs, but will use a new var to not modify self.basis
        else:
            basis = self.basis  # we don't want to fuck with self.basis

        if len(basis) >= 2: # orthonormal must have at least two basis vectors
            for idx, vec1 in enumerate(basis):  # now we run through each basis vector O(n)
                for vec2 in basis[idx+1:]: # we run through each other basis vecor
                    if (dot(vec1, vec2) != 0) or (dot(vec1, vec1) != 1): # if any of them do not pass the test (<v1,v2> = 0 or <v1,v1> =1) (dot defined lower)
                        return False # then we will return False because all must pass the test
                
            return True # otherwise, if False not returned yet, we know it to be true
        
        else: return False # if only one basis vec, nothing to be orthonormal to
    
    def find_projection_mat(self):

        # takes a basis, and assumes it is not columned. it must be columnbed for us to continue.

        if self.columned == False: # if the input basis vecs are row vecs
            basis = mp.transpose(self.basis) # we will make them col vecs, but will use a new var to not modify self.basis
        else:
            basis = self.basis  # we don't want to fuck with self.basis

        if self.isorthonormal: # if the basis vecs r orthonormal, there is a special formula
            return mp.multiply_matrix(basis, mp.transpose(basis)) # we return the projection matrix

        else: # otherwise, we must use the longer forumular
            psueod_inv = mp.multiply_matrix(mp.inverse(mp.multiply_matrix(mp.transpose(basis), basis)), mp.transpose(basis)) # we use da formula O(n**#)
            return mp.multiply_matrix(basis, psueod_inv) # O(n**3)
    
    def vecprojection(self, vector, return_error=False):

        '''
        TAKE:
            self, specifically self.proj_mat
            we also take a vector of the form [x1, ..., xn] where x1,...,xn are real number elements
            return_error flag, automatically set to false, that determines if we should also give the projection_error that occured
        RETURNS:
            the projection of vector onto the subspace represented by this object
        '''

        projected = mp.multiply_matrix(self.proj_mat, [vector]) # the projected coordinates of the vector onto this subspace - note that we must embed the 1d vector into a list to make it 2d

        if return_error == False: # if we dont care abt the projection error
            return projected[0] # we will simply return the projected vector - note the [0] because at this point, projected is a 2d array and the 0th element is the projected vector
        else: # otherwise, if return_error is not False
            return projected[0], euclidean_norm(mp.subtract_row(vector, projected[0])) # we will give back both the projected vector and the projection error

    def make_onb(self, columned=False):

        # we will iteratively implement the Graham Schmidt orthogonalization Al Gore ithm 

        if self.columned == False:
            
        basisvecs = self.basis

        newonb = list()

        for idx, basis in enumerate(basisvectors):
            if idx == 0:
                newonb.append(basis)
                continue

            pmat = find_projection_mat([newonb[-1]])
            new_basis = project_vector(basis, pmat)
            newonb.append(mp.subtract_row(basis,new_basis)) # we do subtract row because these are not matrik, they are individual list
        
        return newonb

class LinearMapping:

  def __init__(self, A, B, mapping=None):
    self.domain_basis = [A[i] for i in mp.identify_pivots(mp.transpose(A))]
    self.domain_dim = len(self.domain_basis)
    self.codomain_basis = [B[i] for i in mp.identify_pivots(mp.transpose(B))]
    self.codomain_dim = len(self.codomain_basis)
    self.domain = [row[:] for row in A]
    self.codomain = [row[:] for row in B]
    self.mapping = mapping # needs to be a matrix or None
    self.injective = True if mp.rank(self.mapping) == len(self.mapping[0]) else False
    self.surjective = True if mp.rank(self.mapping) == len(self.mapping) else False
    self.bijective = True if (self.injective == True and self.surjective == True) else False
    self.homomorphism = True # (by definition, duh)
    self.isomorphism = True if (self.bijective == True) else False # homomorphism already satisfied
    self.endomorphism  = True if len(A) == len(B) and len(A[0]) == len(B[0]) else False # True if dim(A) == dim(B) and len(A[0]) == len(B[0]) else False
    self.automorphism = True if self.endomorphism and self.bijective else False

  def map(self, vector):
        return self.apply(vector)

  def apply_mapping(self, vector):

    # expects a vector in the domain, not codomain
    # expects a vector in the domain, not codomain, of the form  vector = [element1, ...., element 2]
    # will then apply the linear mapping (self.mapping) to the vector
    # will return the result
    if len(vector) != self.domain_dim:
      return "hold up! wait a minute! sumn aint right"

    else:
      return mp.multiply_matrix(self.mapping, mp.transpose([vector]))

  def is_subspace(self, subspace):
      # check if subspace is a part of vector_space_1
      for vector in subspace:
          if vector not in self.matrix:
              return False
      return True

  def typeofmapping(self):
    if self.homomorphism:
      return('it is a homomoprhism')
    elif self.automorphism:
      return('it is a automorphism')
    elif self.endomorphism:
      return('it is a endomorphism')
    elif self.isomorphism:
      return('it is a homomoprhism')
    
    def find_kernel(self):
      pass

#################### FUNCTIONS ####################

def dot(v1, v2):

    if not(isinstance(v1[0], list)) and not(isinstance(v2[0], list)) and (len(v1)==len(v2)):
        total = 0
        for idx in range(len(v1)):
            total += v1[idx]*v2[idx]
        
        return total

def euclidean_norm(vector):

    return (dot(vector, vector))**(1/2)


############## STASH

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