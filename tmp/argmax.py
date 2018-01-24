import numpy as np
#
# a = [[1, 2, 3],
#      [2, 3, 4],
#      [5, 4, 3]]
# print(np.argmax(a, 1))
#
# seq_length = [60 for i in range(20)]
#
# print(seq_length)
a = np.zeros(5)
a[0] = 0
for i in range(1, 5):
    a[i] = i
print(a)
print(len(a))
