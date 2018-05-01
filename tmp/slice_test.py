import numpy as np

a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
b = a[:-2]
c = np.hstack((a[:4], a[5:]))
print(c)
print(b)
print(np.max(a))
