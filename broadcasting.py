import numpy as np

row = np.array([1,2,3])
print(row.shape)
col=np.array([[1],[2],[3]])
print(col.shape)
result=col+row
print(result.shape)
print(result)
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
x = np.array([0.5, 1.0, 2.0])
result=A*x
print(result.shape)
print(result)
b=np.sum(result,axis=1)
print(b)


data = np.random.uniform(-1, 1, size=(5, 5))
print("Original Data:\n", np.round(data, 2))

data[data<=0] = 0

data[data>0.5] = 1

print("\nAfter Activation (ReLU + Gate):\n", data)