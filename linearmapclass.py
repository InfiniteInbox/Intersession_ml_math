class LinearMapping:
  def __init__(self, A, B):
    self.matrix = A
    self.domain = [row[:] for row in A]
    self.codomain = [row[:] for row in B]

  def apply(self, vector):
    # Check that the input vector is in the correct domain
    if len(vector) != len(self.domain[0]):
      raise ValueError("Vector has incorrect dimension for the specified domain")

    # Apply the linear mapping to the input vector
    result = [0 for _ in range(len(self.codomain[0]))]
    for i in range(len(self.matrix)):
      for j in range(len(vector)):
        result[i] += self.matrix[i][j] * vector[j]
    return result

# Test the LinearMapping class
A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]
lm = LinearMapping(A, B)

vector = [1, 2]
result = lm.apply(vector)
print(result)  # Output: [5, 11]