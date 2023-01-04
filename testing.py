
# let a vector be an array v = [v1,...,vn]
# sets of vectors in tuples? probably. so set b = [1, b2,..,bn] where b = [e1,.., en]

# we will need to go from a list of row vectors to column vectors

import matrices as mp

a = [[1,0,4],[12,0,3],[1,0,4]]

a = mp.echelon(a)
print(a)


# will make a class:

class genset:

    def __init__(self, gset):
        self.gset = gset

    def create_basis(self):
        pass
