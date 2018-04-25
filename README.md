# rescal-cpp
Rescal Tensor Factorization using Eigen.


# Dependencies

Eigen: http://eigen.tuxfamily.org/

Pybind11: https://github.com/pybind/pybind11

You can skip Pybind11 if you don't need python bindings but you have to #define NO_PYTHON


# Build

As python module:
```
g++ -O3 -shared -fpic -fopenmp -Wall -std=c++11 -march=native `python -m pybind11 --includes` rescal.cpp -o rescal.so
```


# Usage

From the same directory as rescal.so:
```python
from rescal import *

# Create EdgeList with 5 relations and 100 nodes
X = EdgeList(5, 100)

# Add edges
X.add_edge(r, i, j)
...

# Create Rescal object
rescal = Rescal()

# Run rescal with rank 20
rescal.als(X, 20, lambda_A=.1, lambda_R=.1, max_iter=200)

# Get A matrix
A = rescal.get_A()

# Get 2nd slice of R
R2 = rescal.get_R_slice(1)
```
