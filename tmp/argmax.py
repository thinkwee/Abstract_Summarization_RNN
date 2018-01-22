import numpy as np

a = [[1, 2, 3],
     [2, 3, 4],
     [5, 4, 3]]
print(np.argmax(a, 1))

seq_length = [60 for i in range(20)]

print(seq_length)
