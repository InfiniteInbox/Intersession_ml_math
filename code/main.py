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

import basic_matrix_ops as mp
import copy

#################### GLOBAL VARS ####################

# none

#################### CLASSES ####################

class sub_space:

    '''
    we have created a glass that can somewhat represent a vector subspace
    it works by taking a generating set that will effectively define the subspace
        (note that this is far from perfect and can't truly represent the concept of a vector subspace)
    in the case of being given basis vectors of R^n, it is FACT than the span of those vectors will ALWAYS be a vector space    '''


    def __init__(self, gset, computeoninit=True, columned = False):

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

        if computeoninit == True:


            self.basis = self.create_basis() # we run the self.basis method, which will determine a basis from the given generating set
            self.isorthonormal = self.isorthonormal()
            self.proj_mat = self.find_projection_mat()
            self.onb = make_onb(self.basis, self.columned)
        
        else: 

            self.basis = None
            self.isorthonormal = None
            self.proj_mat = None


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

        return find_projection_mat(self.basis)
    
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

def find_projection_mat(basis, columned=False):

    # takes a basis, and assumes it is not columned. it must be columnbed for us to continue.

    if columned == False:
        basis = mp.transpose(basis)

    # is ortho check reserved for object

    psueod_inv = mp.multiply_matrix(mp.inverse(mp.multiply_matrix(mp.transpose(basis), basis)), mp.transpose(basis))
    return mp.multiply_matrix(basis, psueod_inv)

def project_vector(vector, proj_mat, return_error=False):

    '''
    if using sub_space obj, will look like newvec = project_vector(vec, sub_space.proj_mat)
    '''
    
    projected = mp.multiply_matrix(proj_mat, [vector])

    if return_error == False:
        return projected[0]
    else:
        return mp.mround(projected)[0], euclidean_norm(mp.subtract_row(vector, projected[0]))

def proj_to_affine(basis_vectors, vec_to_proj, offset):

    pmat = find_projection_mat(basis_vectors)
    newvec_wo_offset = project_vector(vec_to_proj, pmat)

    return mp.subtract_rows(newvec_wo_offset, mp.row_by_scalar(offset, -1))

def make_onb(basisvectors, columned=False):
    # we will iteratively implement the Graham Schmidt orthogonalization Al Gore ithm 

    newonb = list()

    for idx, basis in enumerate(basisvectors): 
        if idx == 0:
            newonb.append(basis)
            continue    

        pmat = find_projection_mat([newonb[-1]])
        new_basis = project_vector(basis, pmat)
        newonb.append(mp.subtract_row(basis,new_basis)) # we do subtract row because these are not matrik, they are individual list
    
    return newonb

def isuppertriangle(matrix, tolerance=6):

    for idx, row in enumerate(matrix):
        
        for idx2 in range(idx):
            if round(row[idx2], tolerance) != 0:
                return False
    
    return True

def qrdecomp(matrix):

    if mp.rank(matrix) != len(matrix):
        raise ValueError("Matrix is rank deficient and cannot be fully QR Decomposed")
    
    matrix = mp.transpose(matrix)

    q = list()
    r = list()
    uactive = None
    ulist = list()
    for idx, column in enumerate(matrix):
        if idx == 0:
            uactive = column
        else:
            uactive = column
            for oldu in ulist:

                inprodscalar = ((dot(oldu, column)) / (euclidean_norm(oldu))**2)
                projected = mp.row_by_scalar(oldu, inprodscalar)
                uactive = mp.subtract_row(uactive, projected)

        

        norm = 1/(euclidean_norm(uactive))
        e = mp.row_by_scalar(uactive, norm)

        ulist.append(uactive)
        q.append(e)

    q = mp.transpose(q)
    qinv = mp.inverse(q.copy())
    r = mp.multiply_matrix(qinv,mp.transpose(matrix))
    return q,r 

def eigvals(matrix, iterations = 1000, tolerance=6):

    activemat = copy.deepcopy(matrix)

    istriangle = False

    for i in range(iterations):
    # while not istriangle:
        q, r = qrdecomp(activemat)
        activemat = mp.multiply_matrix(r,q)

            # istriangle = isuppertriangle(activemat, tolerance)


    # g = isuppertriangle(activemat, tolerance)
    # return activemat

    if isuppertriangle(activemat, tolerance):

        return mp.diags(activemat)


    else:

        eigvals = mp.diags(activemat)

        for idx in range(len(activemat)-1):

            if round(activemat[idx+1][idx], 8) != 0:

                a = activemat[idx][idx]
                b = activemat[idx][idx+1]
                c = activemat[idx+1][idx]
                d = activemat[idx+1][idx+1]

                term1 = -(-a-d)
                term2 = ((term1**2) - 4*(a*d - b*c))
                positiveroot = (term1 + term2**(1/2))/2
                negativeroot = (term1 - term2**(1/2))/2


                del eigvals[idx:idx+2]
                eigvals.insert(idx,positiveroot)
                eigvals.insert(idx+1,negativeroot)

                idx+=1

        return eigvals

def eigvecs(matrix, eigvalitr = 1000, eigvaltol = 6, vec_tol = 4):

    eigval_list = eigvals(matrix, eigvalitr, eigvaltol)
    matrix_size = len(matrix)
    eigvec_dict = {}

    for eigval in eigval_list:
        
        if isinstance(eigval, complex):
            eigval = (round(eigval.real, eigvaltol) + round(eigval.imag, eigvaltol)*1j)
        else:
            eigval = round(eigval, eigvaltol)
        idt = mp.make_identity(matrix_size)
        idt = mp.matrix_by_scalar(idt, eigval)
        subtracted_mat = mp.subtract_matrices(matrix, idt)
        # subtracted_mat = mp.mround(subtracted_mat, vec_tol)
        subtracted_mat = mp.echelon(subtracted_mat)
        subtracted_mat = mp.mround(subtracted_mat, vec_tol)

        sln = mp.solve_homogeneous(subtracted_mat)
    
        eigvec_dict[eigval] = sln
    

    return eigvec_dict

def eigendecomp(matrix, eigvalitr = 1000, eigvaltol = 6, vec_tol = 4, normalize=True):

    eigvec_dict = eigvecs(matrix, eigvalitr, eigvaltol, vec_tol)

    p = list()
    d = list()

    i = 0
    for eigval, eigvec in eigvec_dict.items():

        if normalize == True:
            scalar = euclidean_norm(eigvec[0])
            eigvec = mp.matrix_by_scalar(eigvec, (1/scalar))

        d.append([0 if idx != i else eigval for idx in range(len(matrix))])
        p.append(eigvec[0])
        i += 1

    p = mp.transpose(p)
    pinv = mp.inverse(p)
    
    return p, d, pinv

def svd(matrix, eigvalitr = 1000, eigvaltol = 6, vec_tol = 4, normalize=True):

    v, d, useless = eigendecomp((mp.multiply_matrix(mp.transpose(matrix), matrix)), normalize)
    # u = eigendecomp((mp.multiply_matrix(a, mp.transpose(a))), normalize)[0]

    v = mp.matrix_by_scalar(v, -1)
    u = list()
    sig = list()

    for idx in range(len(matrix)):
        sig.append([0 if idx != i else ((d[idx][idx])**(1/2)) for i in range(len(d[idx]))])
    
    for idx in range(len(sig)):
        u_idx = mp.multiply_matrix(matrix, mp.transpose([mp.get_col(v, idx)]))
        u_idx = mp.matrix_by_scalar(u_idx, 1/sig[idx][idx])
        u.append(mp.transpose(u_idx)[0])
    u = mp.transpose(u)

    return u, sig, mp.transpose(v)


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
        og_base, new_base = mp.transpose(og_base), mp.transpose(new_base)

    # we find a transition matrix that changes the coordinates expressed from one base to another base

    # going from base 1 to base 2
    if len(og_base) == len(new_base) and len(og_base[0]) == len(new_base[0]):

        dim_to_take = len(og_base[0]) # we find what the dimensions of the transmat will be
        total = mp.append_mat_right(new_base, og_base) # we create an augmented matrix with new basis on left and old basis on right
        reduced = mp.rref(total) # row reduce, O(n**3)

        transition_matrix = list() # init our blank transition matrix

        for row in reduced: # we go through the reduced augmat to find what should be added to the transition matrix
            transition_matrix.append(row[-dim_to_take:])

        return transition_matrix

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
        ogb1, ogb2, tildab1, tildab2 = mp.transpose(ogb1),mp.transpose(ogb2),mp.transpose(tildab1),mp.transpose(tildab2) 

    transition1 = find_transition(tildab1, ogb1) # we find transitino matrix from the new domain basis to the old domain basis, O(n**3)
    transition2 = mp.inverse(find_transition(tildab2, ogb2)) # we find inverse of transition matrix from new codomain basis to old codomain basis, O(n**3)

    return mp.multiply_matrix(mp.multiply_matrix(transition2, ogtransfmat), transition1) # we multiply the matrices according to formula, O(n**3)

