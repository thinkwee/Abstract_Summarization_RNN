import numpy as np

a = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
b = a[:-2]
c = np.hstack((a[:4], a[5:]))
print(c)
print(b)
print(np.max(a))
print(a[:5])
d = []
for i in range(0, len(a), 3):
    d.append(a[i:i + 3])
print(d)