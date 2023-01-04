
# let a vector be an array v = [v1,...,vn]
# sets of vectors in tuples? probably. so set b = [1, b2,..,bn] where b = [e1,.., en]

# we will need to go from a list of row vectors to column vectors


def get_col(matrix_2d, _index):

    return list(row[_index] for row in matrix_2d) # O(n) this simply grabs the column from the specified index. 

def transpose(matrix):

    new_array = [get_col(matrix, i) for i in range(len(matrix[0]))] # O(n) and nested O(n), becomes O(n**2). takes a column and makes it a row
    return new_array

# will make a class:

class genset:

    def __init__(self, gset):
        self.gset = gset

    def create_basis(self):
        pass
