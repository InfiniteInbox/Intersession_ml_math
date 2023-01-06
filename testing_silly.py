# mindim = 3

# for idx in reversed(range(mindim)):
#     # if matrix[idx][idx] == 1:
#         for idx2 in reversed(range(idx)):
#             print(f'{idx}, {idx2}')

l = [0,1,2,3,4]

for i in range(len(l)):
    for j in l[i+1:]:
        print(f"{i}, {j}")