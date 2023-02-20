## Research

Running minimize the spec or implementation for looking in benchmarks.

```python 
import numpy as np

def gauss_elim(A, b):
    n = len(A)
    for i in range(n):
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j] -= factor * A[i]
            b[j] -= factor * b[i]
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i][i+1:], x[i+1:])) / A[i][i]
    return x

A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])
x = gauss_elim(A, b)
print(x)

```
How to run: query the AI

Thread: query AI in multi AI connect if non trivial? 
