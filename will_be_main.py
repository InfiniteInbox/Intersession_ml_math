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

#################### GLOBALS ####################

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

        if len(basis) >= 2: # orthonormal must have at least two basis vectors
            for idx, vec1 in enumerate(basis):  # now we run through each basis vector O(n)
                for vec2 in basis[idx+1:]: # we run through each other basis vecor
                    if (dot(vec1, vec2) != 0) or (dot(vec1, vec1) != 1): # if any of them do not pass the test (<v1,v2> = 0 or <v1,v1> =1) (dot defined lower)
                        return False # then we will return False because all must pass the test
                
            return True # otherwise, if False not returned yet, we know it to be true
        
        else: return False # if only one basis vec, nothing to be orthonormal to

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
