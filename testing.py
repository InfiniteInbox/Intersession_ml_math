
# let a vector be an array v = [v1,...,vn]
# sets of vectors in tuples? probably. so set b = [1, b2,..,bn] where b = [e1,.., en]

# we will need to go from a list of row vectors to column vectors

import matrices as mp

# a = [[1,0,0],[0,1,0],[0,0,0]]

# a = mp.echelon(a)
# print(a)
# print(mp.identify_pivots(a))


# will make a class:

class genset:

    def __init__(self, gset, columned = False):
        self.gset = gset
        self.columned = columned

    def create_basis(self):

        tempgset = self.gset

        if self.columned == False:
            tempgset = mp.transpose(tempgset)
        
        holder = mp.identify_pivots(tempgset)

        return [self.gset[i] for i in holder]
    
def dot(v1, v2):

    if not(isinstance(v1[0], list)) and not(isinstance(v2[0], list)) and (len(v1)==len(v2)):
        total = 0
        for idx in range(len(v1)):
            total += v1[idx]*v2[idx]
        
        return total

def euclidean_norm(vector):

    return (dot(vector, vector))**(1/2)

def isortho(v1,v2):

    # inner op is dot product
    if dot(v1,v2) == 0:
        return True
    else:
        return False 

def isorthonormal(basis, columned=False): 

    # takes a matrix of matrices, where each inner matrix is a basis vector. assumed not columned, which ultimately is what we want for this

    if len(basis) >= 2:
        for idx, vec1 in enumerate(basis):
            for vec2 in basis[idx+1:]:
                if (dot(vec1, vec2) != 0) or (dot(vec1, vec1) != 1):
                    return False
            
        return True
    
    else: return False

def find_projection_mat(basis, columned=False):

    # takes a basis, and assumes it is not columned. it must be columnbed for us to continue.

    isortho = isorthonormal(basis)

    if columned == False:
        basis = mp.transpose(basis)
    
    if isortho:
        return mp.multiply_matrix(basis, mp.transpose(basis))

    else:
        psueod_inv = mp.multiply_matrix(mp.inverse(mp.multiply_matrix(mp.transpose(basis), basis)), mp.transpose(basis))
        return mp.multiply_matrix(basis, psueod_inv)

def project_vector(vector, proj_mat, return_error=False):
    
    projected = mp.multiply_matrix(proj_mat, [vector])

    if return_error == False:
        return projected[0]
    else:
        return mp.mround(projected)[0], euclidean_norm(mp.subtract_row(vector, projected[0]))

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

def proj_to_affine(basis_vectors, vec_to_proj, offset):

    pmat = find_projection_mat(basis_vectors)
    newvec_wo_offset = project_vector(vec_to_proj, pmat)

    return mp.subtract_rows(newvec_wo_offset, mp.row_by_scalar(offset, -1))

print(make_onb([[2,0], [1,1]]))

# a = [[1,1,1], [0,1,2]]
# psuba = find_projection_mat(a)

# print(project_vector([6,0,0], psuba, return_error = True))

# # a = [[0,-1,2,0,2], [1,-3,1,-1,2], [-3,4,1,2,1]]
# a = [[2,0]]

# b = find_projection_mat(a)
# print(b)
# c = project_vector([1,1],b)
# print(c)
# a = [
#     [3,2],
#     [1,4],
#     [2,4]    ]

# b = [
#     [1],
#     [2]]

# print(mp.multiply_matrix(a, b))
    

# g = genset([[1,2,-1,-1,-1], [2,-1,1,2,-2], [3,-4,3,5,-3], [-1,8,-5,-6,1]])

# print(g.get_rank())

# a = [
#     [94,3,9],
#     [24,7,18],
#     [38,2,10]]

# b = [
#     [1,0,0],
#     [0,1,0],
#     [0,0,1]
#     ]

# # h = mp.append_mat(b, a)
# # h = mp.rref(h)

# a1 = [[1,2,0],[-1,1,3],[3,7,1],[-1,2,4]]
# b1 = [[1,0,0],[0,1,0],[0,0,1]]
# c1 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
# b_new = [[1,0,1],[1,1,0],[0,1,1]]
# c_new = [[1,1,0,1],[1,0,1,0],[0,1,1,0],[0,0,0,1]]

# print(mp.change_transformation_basis(a1, b1, c1, b_new, c_new))