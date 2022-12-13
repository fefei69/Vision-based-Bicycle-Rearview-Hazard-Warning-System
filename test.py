from collections import deque

pt = [deque(maxlen=50) for _ in range(100)]
id = 0
numbers = (1,2,3)
pt[id].append(numbers)
print(pt)
print(pt[0][0])
print(type(pt))