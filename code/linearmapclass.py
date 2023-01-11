import matrices as mp

class genset:

    def __init__(self, gset, columned = False):
        self.gset = gset
        self.columned = columned

    def create_basis(self):

        tempgset = self.gset

        if self.columned == False:
            tempgset = mp.transpose(tempgset)
        
        holder = mp.identify_pivots(tempgset)
        print(holder)

        return [self.gset[i] for i in holder]

t1 = genset([[1,2,-1,-1,-1], [2,-1,1,2,-2], [3,-4,3,5,-3], [-1,8,-5,-6,1]])
print(t1.create_basis())

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

# Test the LinearMapping class
# A = [[1, 2], [3, 4]]
# B = [[5, 6, 7], [8, 9, 10],[11, 12, 13]]
# C = [[1,3],[2,3],[2,4]]
# lm = LinearMapping(A, B, mapping=C)
# xd = lm.typeofmapping()
# print(xd)
# subspac1e = [1,2]
# result = lm.apply_mapping(subspac1e)
# print(result)