# Sparse Matrices

Efficiently computes the cholesky factorization on a sparse matrix. I created this as an application of my undergraduate thesis in Math, as such its not really meant to be used.

```
std::vector<int> ti = {0, 1, 2, 1, 3, 2, 3, 3, 4, 4}; // row indices
std::vector<int> tj = {0, 0, 0, 1, 1, 2, 2, 3, 3, 4}; // col indices
std::vector<double> tx = {5, 1, 1, 4, 1, 4, 1, 5, 1, 3}; // values

const int n = 5;

auto A = triplet_to_csc_matrix(ti, tj, tx, n);

std::cout << to_string(A) << std::endl;
```

```
> 5  1  1  0  0
  1  4  0  1  0
  1  0  4  1  0
  0  1  1  5  1
  0  0  0  1  3
```

```
auto S = schol(A); // symbolic
auto L = chol(A, S).value().value(); // numeric

std::cout << to_string(L) << "\n";
```

```
> 2.24  0.45   0.45  0    0
  0.45  1.95  -0.10  0.51 0
  0.45  -0.10 1.95   0.54 0
  0     0.51  0.54   2.11 0.47
  0     0     0      0.47 1.67
```
