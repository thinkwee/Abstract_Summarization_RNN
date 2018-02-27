import numpy as np

a = [1, 2, 3, 4, 5, 6]
b = [1, 2, 3]

a = np.resize(a, [3, 5])

print(a)
print(a + b)
