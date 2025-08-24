#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>
#include <vector>

/// Symmetric positive definite (SPD) matrix in compressed sparse row (CSR) format
template <typename T>
class spd {
private:
	/// Matrix is n x n (square)
	std::size_t n_ = 0;

	/// Max non-zero entries (lower triangle only)
	std::size_t nzmax_ = 0;

	/// Row pointers (size n+1), like CSR
	std::vector<int> p_;

	/// Column indices (size nzmax)
	std::vector<int> j_;

	/// Non-zero values (size nzmax)
	std::vector<T> x_;

public:
	/// Constructor
	spd(int n, int nzmax)
		: n_(n)
		, nzmax_(nzmax)
		, p_(n + 1, 0)
		, j_(nzmax)
		, x_(nzmax, 0.0) {

		// Check that n and nzmax are positive
		assert(n > 0 && nzmax > 0);

		// Check that nzmax is at most the number of entries in lower triangle
    	assert(nzmax <= n * (n + 1) / 2);
	}

	// =============
	//   Accessors
	// =============
	int size() const {
		return n_;
	}
	int capacity() const {
		return nzmax_;
	}

	std::vector<int>& p() {
		return p_;
	}
	std::vector<int>& j() {
		return j_;
	}
	std::vector<T>& x() {
		return x_;
	}

	const std::vector<int>& p() const {
		return p_;
	}
	const std::vector<int>& j() const {
		return j_;
	}
	const std::vector<T>& x() const {
		return x_;
	}

	// Read-only element access (i, j)
	double operator[](int i, int j) const {
		assert(i >= 0 && i < n_ && j >= 0 && j < n_);
		if(j > i)
			std::swap(i, j); // Only store lower triangle

		for(int idx = p_[i]; idx < p_[i + 1]; ++idx) {
			if(j_[idx] == j) {
				return x_[idx];
			}
		}
		return 0.0; // Implicit zero if not stored
	}
};

/// Convert triplet format to symmetric positive definite (SPD) matrix
template <typename T>
spd<T> triplet_to_spd(const std::vector<int>& ti, // row indices
					  const std::vector<int>& tj, // column indices
					  const std::vector<T>& tx, // values
					  int n) {
	assert(ti.size() == tj.size() && tj.size() == tx.size());

	using Entry = std::tuple<int, int, T>; // (row, col, value)
	std::vector<Entry> entries;

	// Only store lower triangle (i >= j)
	for(std::size_t k = 0; k < ti.size(); ++k) {
		int i = ti[k], j = tj[k];
		double val = tx[k];
		assert(i >= 0 && i < n && j >= 0 && j < n);
		if(j > i)
			std::swap(i, j);
		entries.emplace_back(i, j, val);
	}

	// Sort by (i, j) to group duplicates and make rows contiguous
	std::sort(entries.begin(), entries.end());

	// Combine duplicates and count non-zeros per row
	std::vector<int> row_nnz(n, 0);
	std::vector<Entry> merged;
	for(std::size_t k = 0; k < entries.size(); ++k) {
		if(!merged.empty() && std::get<0>(merged.back()) == std::get<0>(entries[k]) &&
		   std::get<1>(merged.back()) == std::get<1>(entries[k])) {
			std::get<2>(merged.back()) += std::get<2>(entries[k]); // Combine duplicate
		} else {
			merged.push_back(entries[k]);
			row_nnz[std::get<0>(entries[k])]++;
		}
	}

	// Exclusive prefix sum to build p_
	std::vector<int> p(n + 1, 0);
	for(int i = 0; i < n; ++i)
		p[i + 1] = p[i] + row_nnz[i];

	int nzmax = merged.size();
	spd<T> A(n, nzmax);
	auto& j = A.j();
	auto& x = A.x();
	A.p() = std::move(p); // Already constructed

	// Fill j and x
	std::vector<int> offset = A.p(); // Track insertion point per row
	for(const auto& [i, col, val] : merged) {
		int pos = offset[i]++;
		j[pos] = col;
		x[pos] = val;
	}

	return A;
}

/// Compute elimination tree of symmetric SPD matrix stored in lower-triangular CSR
template <typename T>
std::vector<int> etree(const spd<T>& A) {
    int n = A.size();
    const auto& p = A.p();
    const auto& j = A.j();

    std::vector<int> parent(n, -1);
    std::vector<int> ancestor(n, -1);

    for (int i = 0; i < n; i++) {
        // row i
        for (int idx = p[i]; idx < p[i+1]; idx++) {
            int col = j[idx];

			// skip diagonal and above
			// col < i
            if (col >= i) 
				continue;

            int k = col;
            while (k != -1 && k < i) {
                int next = ancestor[k];
                ancestor[k] = i; // path compression
                if (next == -1) {
                    parent[k] = i; // found parent
                }
                k = next;
            }
        }
    }
    return parent;
}
