
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
        print(holder)

        return [self.gset[i] for i in holder]

g = genset([[1,2,-1,-1,-1], [2,-1,1,2,-2], [3,-4,3,5,-3], [-1,8,-5,-6,1]])

print(g.create_basis())

# a = [[1,2,-1,-1,-1], [2,-1,1,2,-2], [3,-4,3,5,-3], [-1,8,-5,-6,1]]
# a = mp.transpose(a)
# a = mp.ref(a)
# print(a)