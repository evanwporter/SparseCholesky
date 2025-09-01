#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <expected>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {
#include <cblas.h>

/// Computes the Cholesky factorization of a real symmetric positive definite matrix A.
void dpotrf_(const char* uplo, const int* n, double* a, const int* lda, int* info);
}

#include "pcg_random.hpp"

enum class sym {
    none,
    upper,
    lower
};

using elimination_tree = std::vector<int>;

struct SChol {
    /// Matrix size (rows/cols)
    std::size_t n_ = 0;

    /// Elimination tree (size n)
    std::vector<int> parent;

    /// Column pointers for L (size n+1)
    std::vector<int> cp;

    /// Row indices for L (size nnz = cp.back())
    std::vector<int> rowind;

    /// Column pointers for L (size n+1)
    std::vector<int> p_;

    /// Row indices for L (size nnz = cp.back())
    std::vector<int> i_;

    /// Number nonzeros in lower triangle
    std::size_t nnz = 0;

    /// Number nonzeros in lower triangle
    std::size_t lnz = 0;

    /// Number nonzeros in upper triangle
    std::size_t unz = 0;

    // Access (i, j) with symmetry awareness
    bool operator[](int i, int j) const {
        assert(i >= 0 && i < n_ && j >= 0 && j < n_);

        if (i < j)
            std::swap(i, j);

        for (int idx = p_[j]; idx < p_[j + 1]; ++idx) {
            if (i_[idx] == i)
                return true;
        }

        // If we can't find it in the column pointers
        return false;
    }

    size_t rows() const { return n_; }
    size_t cols() const { return n_; }

    size_t size() const { return n_; }
    size_t capacity() const { return nnz; }

    std::vector<int>& p() { return p_; }
    std::vector<int>& i() { return i_; }

    const std::vector<int>& p() const { return p_; }
    const std::vector<int>& i() const { return i_; }
};

template <typename T, sym S = sym::upper>
class csc_matrix {
private:
    /// Rows
    std::size_t m_ = 0;

    // Cols
    std::size_t n_ = 0;

    /// Max non-zero entries
    std::size_t nzmax_ = 0;

    /// Column pointers (size n+1)
    // Contains the indice where the value starts in j_ and x_
    std::vector<int> p_;

    /// Row indices (size nzmax)
    std::vector<int> i_;

    /// Non-zero values (size nzmax)
    std::vector<T> x_;

    T* find_entry(std::size_t i, std::size_t j) {
        if constexpr (S == sym::upper) {
            if (j < i)
                std::swap(i, j);
        } else if constexpr (S == sym::lower) {
            if (i < j)
                std::swap(i, j);
        }

        auto col_start = p_[j];
        auto col_end = p_[j + 1];

        // binary search to find the row
        auto it = std::lower_bound(i_.begin() + col_start, i_.begin() + col_end, static_cast<int>(i));
        if (it != i_.begin() + col_end && *it == static_cast<int>(i)) {
            auto idx = std::distance(i_.begin(), it);
            return &x_[idx];
        }

        return nullptr;
    }

public:
    csc_matrix(std::size_t m, std::size_t n, int nzmax) :
        n_(n),
        m_(m),
        nzmax_(nzmax),
        p_(n + 1, 0),
        i_(nzmax),
        x_(nzmax, T(0)) {

        assert(m > 0 && n > 0 && nzmax > 0);

        if constexpr (S != sym::none) {

            // enforce square for symmetric
            assert(m == n);
        }
    }

    /// Construct CSC matrix directly from symbolic analysis
    csc_matrix(const SChol& schol)
        requires(S != sym::none)
        :
        m_(schol.parent.size()),
        n_(schol.parent.size()),
        nzmax_(schol.cp.back()),
        p_(schol.cp),
        i_(schol.rowind),
        x_(schol.cp.back(), T(0)) {
        assert(m_ == n_);
    }

    // Constructor only available if Symmetric
    csc_matrix(std::size_t n, std::size_t nzmax)
        requires(S != sym::none)
        :
        m_(n), n_(n), nzmax_(nzmax),
        p_(n + 1, 0), i_(nzmax), x_(nzmax, T(0)) {
        assert(nzmax <= n * (n + 1) / 2);
        assert(n > 0 && nzmax > 0);
    }

    size_t rows() const { return m_; }
    size_t cols() const { return n_; }

    size_t size() const { return n_; }
    size_t capacity() const { return nzmax_; }

    std::vector<int>& p() { return p_; }
    std::vector<int>& i() { return i_; }
    std::vector<T>& x() { return x_; }

    const std::vector<int>& p() const { return p_; }
    const std::vector<int>& i() const { return i_; }
    const std::vector<T>& x() const { return x_; }

    // Access (i, j) with symmetry awareness.
    T& operator[](std::size_t i, std::size_t j) {
        assert(i < m_ && j < n_);
        auto* ptr = find_entry(i, j);
        if (!ptr) {
            throw std::out_of_range("Element not present in CSC structure");
        }
        return *ptr;
    }

    // Access (i, j) with symmetry awareness.
    const T operator[](std::size_t i, std::size_t j) const {
        assert(i < m_ && j < n_);
        auto* ptr = const_cast<csc_matrix*>(this)->find_entry(i, j);
        if (!ptr) {
            return T(0);
        }
        return *ptr;
    }

    /// Transpose matrix
    csc_matrix<T, sym::none> transpose() const {
        const int m = static_cast<int>(m_);
        const int n = static_cast<int>(n_);
        const auto& Ap = p_;
        const auto& Ai = i_;
        const auto& Ax = x_;

        // Allocate result arrays
        std::vector<int> ATp(m + 1, 0);
        std::vector<int> ATi(Ap.back(), 0);
        std::vector<T> ATx(Ap.back(), T(0));

        // Count entries per row (future AT columns)
        for (int j = 0; j < n; ++j) {
            for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
                int r = Ai[p];
                ATp[r + 1]++;
            }
        }

        // Cumulative sum -> ATp
        for (int r = 0; r < m; ++r)
            ATp[r + 1] += ATp[r];

        // Fill ATi and ATx
        std::vector<int> next = ATp; // rolling write heads
        for (int j = 0; j < n; ++j) {
            for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
                int r = Ai[p];
                int dest = next[r]++;
                ATi[dest] = j; // col index becomes row index
                ATx[dest] = Ax[p]; // copy value
            }
        }

        // Build result matrix
        csc_matrix<T, sym::none> AT(m, n, static_cast<int>(ATi.size()));
        AT.p() = std::move(ATp);
        AT.i() = std::move(ATi);
        AT.x() = std::move(ATx);

        return AT;
    }
};

/**
 * @brief Convert triplet format to symmetric positive definite (SPD) matrix
 * @param ti row indeices
 * @param tj column indices
 * @param tx values
 * @param n size of matrix (`nxn`)
 */
template <typename T>
csc_matrix<T, sym::upper> triplet_to_csc_matrix(const std::vector<int>& ti, const std::vector<int>& tj, const std::vector<T>& tx, int n) {
    assert(ti.size() == tj.size() && tj.size() == tx.size());

    using Entry = std::tuple<int, int, T>; // (row, col, value)

    std::vector<Entry> entries;

    // Only store upper triangle (i >= j)
    for (std::size_t k = 0; k < ti.size(); ++k) {
        int i = ti[k], j = tj[k];
        double val = tx[k];
        assert(i >= 0 && i < n && j >= 0 && j < n);
        if (j < i)
            std::swap(i, j);
        entries.emplace_back(i, j, val);
    }

    // Sort by (j, i) to group duplicates and make rows contiguous
    std::sort(entries.begin(), entries.end(), [](auto& a, auto& b) {
        if (std::get<1>(a) != std::get<1>(b))
            return std::get<1>(a) < std::get<1>(b);
        return std::get<0>(a) < std::get<0>(b);
    });

    // Merge duplicates and count per col nonzeros
    std::vector<int> col_nnz(n, 0);
    std::vector<Entry> merged;
    for (auto& e : entries) {
        int r, c;
        T v;
        std::tie(r, c, v) = e;
        if (!merged.empty() && std::get<0>(merged.back()) == r && std::get<1>(merged.back()) == c) {
            std::get<2>(merged.back()) += v;
        } else {
            merged.push_back(e);
            col_nnz[c]++;
        }
    }

    /// Col pointers
    std::vector<int> p(n + 1, 0);
    for (int j = 0; j < n; j++)
        p[j + 1] = p[j] + col_nnz[j];

    csc_matrix<T, sym::upper> A(n, static_cast<int>(merged.size()));
    A.p() = p;
    auto& Ai = A.i();
    auto& Ax = A.x();

    std::vector<int> offset = A.p();
    for (auto& e : merged) {
        int r, c;
        T v;
        std::tie(r, c, v) = e;
        int pos = offset[c]++;
        Ai[pos] = r;
        Ax[pos] = v;
    }

    return A;
}

/**
 * @brief Compute the elimination tree of an SPD matrix.
 *
 * @param A SPD matrix to compute the elimination tree of.
 * @return elimination_tree size n, where `parent[j]` is the parent of column `j`
 */
template <typename T>
elimination_tree etree(const csc_matrix<T, sym::upper>& A) {
    const int n = static_cast<int>(A.size());
    const auto& Ap = A.p();
    const auto& Ai = A.i();

    elimination_tree parent(n, -1);
    std::vector<int> ancestor(n, -1);

    for (int k = 0; k < n; k++) {
        parent[k] = -1; // no parent yet
        ancestor[k] = -1; // no ancestor yet

        for (int p = Ap[k]; p < Ap[k + 1]; p++) {
            int i = Ai[p]; // row index of entry (i,k)
            if (i > k)
                continue; // only use upper triangle

            // traverse ancestors of i up to k
            while (i != -1 && i < k) {
                int inext = ancestor[i];
                ancestor[i] = k; // path compression

                if (inext == -1) {
                    parent[i] = k; // found new parent
                    break;
                }

                i = inext;
            }
        }
    }
    return parent;
}

// TODO clarify name
/// Build SPD from a "row pattern" input (each row lists the columns to include).
/// Only upper triangle (i <= j) entries are stored. Values are set to 1 by default.
template <typename T>
csc_matrix<T, sym::upper> build_csc_matrix_from_pattern(const std::vector<std::vector<int>>& pattern) {
    std::vector<int> ti;
    std::vector<int> tj;
    std::vector<T> tx;

    const int n = static_cast<int>(pattern.size());

    for (int i = 0; i < n; ++i) {
        for (int col : pattern[i]) {
            int r = i, c = col;
            if (c < r)
                std::swap(r, c); // enforce upper triangle
            ti.push_back(r);
            tj.push_back(c);
            tx.push_back(T(1)); // default value = 1
        }
    }

    return triplet_to_csc_matrix(ti, tj, tx, n);
}

/**
 * @brief Iterative DFS used by postorder
 * @note Taken from SuiteSparse (Dr. Tim Davis)
 *
 * @param next sibling links
 * @param post output post order
 * @param stack temporary stach (size >= n)
 */
static int tdfs(int root, int k, std::vector<int>& head, const std::vector<int>& next, std::vector<int>& post, std::vector<int>& stack) {
    int top = 0;
    stack[0] = root;
    while (top >= 0) {
        int p = stack[top]; // node on top
        int child = head[p]; // youngest (unvisited) child
        if (child == -1) { // no more children -> finish p
            top--;
            post[k++] = p;
        } else { // descend to child
            head[p] = next[child]; // pop child from p's list
            stack[++top] = child;
        }
    }
    return k;
}

/**
 * @brief Postorder of a forest given parent[] (parent[j] == -1 for roots).
 * @note Taken from SuiteSparse (Dr. Tim Davis)
 */
inline std::vector<int> post_order(const elimination_tree& parent) {
    const int n = static_cast<int>(parent.size());
    assert(n >= 0);

    std::vector<int> post(n, -1);

    /// head[v] = youngest child of v (singly-linked list)
    std::vector<int> head(n, -1);

    /// next[s] = older sibling of s (in the list of parent[s])
    std::vector<int> next(n, -1);

    /// DFS stack
    std::vector<int> stack(n, -1);

    // Build child lists so that siblings are visited in the same order as cs_post:
    // traverse j from n-1 down to 0, push j to front of parent[j]'s list.
    for (int j = n - 1; j >= 0; --j) {
        int p = parent[j];
        if (p == -1)
            continue; // j is a root
        next[j] = head[p];
        head[p] = j;
    }

    // Postorder each tree
    int k = 0;
    for (int j = 0; j < n; ++j) {
        if (parent[j] != -1)
            continue; // skip non-roots
        k = tdfs(j, k, head, next, post, stack);
    }
    return post; // size n, a permutation of 0..n-1
}

/**
 * @brief Transpose (pattern only) of the matrix `A`
 *
 * @param A matrix to be transposed
 */
template <typename T>
static void transpose_pattern(const csc_matrix<T>& A, std::vector<int>& ATp, std::vector<int>& ATi) {
    const int n = static_cast<int>(A.size());
    const auto& Ap = A.p();
    const auto& Ai = A.i();

    ATp.assign(n + 1, 0);
    ATi.assign(Ap.back(), 0);

    // Count entries per row (== per AT column)
    for (int j = 0; j < n; ++j) {
        for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
            int r = Ai[p];
            ATp[r + 1]++; // one entry goes to column r in AT
        }
    }
    // Cumulative sum → ATp
    for (int j = 0; j < n; ++j)
        ATp[j + 1] += ATp[j];

    // Fill ATi with column indices (original col j becomes row j in AT)
    std::vector<int> next = ATp; // rolling write heads
    for (int j = 0; j < n; ++j) {
        for (int p = Ap[j]; p < Ap[j + 1]; ++p) {
            int r = Ai[p];
            ATi[next[r]++] = j; // AT(r, j) = A(j, r)
        }
    }
}

// Edge processor
static inline void process_edge(
    int j, int i, const std::vector<int>& first, std::vector<int>& maxfirst, std::vector<int>& delta, std::vector<int>& prevleaf, std::vector<int>& ancestor) {
    if (i <= j || first[j] <= maxfirst[i])
        return;

    maxfirst[i] = first[j]; // update max first[j] seen for row i
    int jprev = prevleaf[i]; // previous leaf in subtree i
    delta[j]++; // A(i,j) contributes to skeleton

    if (jprev != -1) {
        // Least common ancestor (LCA) of jprev and j with path compression
        int q = jprev;
        while (q != ancestor[q])
            q = ancestor[q];
        for (int s = jprev; s != q;) {
            int sparent = ancestor[s];
            ancestor[s] = q;
            s = sparent;
        }
        delta[q]--; // remove overlap at the LCA
    }
    prevleaf[i] = j;
}

/**
 * @brief Column counts for Cholesky (ata = 0 path of cs_counts)
 *
 * @return `colcount[j] = nnz` in column `j` of `L` (incl. diagonal)
 */
template <typename T>
std::vector<int> col_count(const csc_matrix<T>& A, const std::vector<int>& parent, const std::vector<int>& post) {
    const int n = static_cast<int>(A.size());
    assert(static_cast<int>(parent.size()) == n);
    assert(static_cast<int>(post.size()) == n);

    // Transpose to iterate rows more quickly
    std::vector<int> ATp, ATi;
    transpose_pattern(A, ATp, ATi);

    // Workspace
    std::vector<int> colcount(n, 0);
    std::vector<int> first(n, -1);
    std::vector<int> maxfirst(n, -1);
    std::vector<int> prevleaf(n, -1);
    std::vector<int> ancestor(n);
    for (int i = 0; i < n; ++i)
        ancestor[i] = i;

    // Initialize first[] and delta[] as in cs_counts
    // Scan nodes in postorder index k = 0..n-1
    std::vector<int> delta = colcount; // reuse storage pattern; will overwrite
    for (int k = 0; k < n; ++k) {
        int j = post[k];
        delta[j] = (first[j] == -1) ? 1 : 0; // j is a leaf if first[j] was unset
        for (; j != -1 && first[j] == -1; j = parent[j]) {
            first[j] = k;
        }
    }

    // Main pass: process each node in postorder
    for (int k = 0; k < n; ++k) {
        int j = post[k]; // kth node in postordered etree

        if (parent[j] != -1)
            delta[parent[j]]--; // j is not a root

        // Iterate "row j" via AT: all i with A(j,i) ≠ 0 (note: i >= j in your storage)
        for (int p = ATp[j]; p < ATp[j + 1]; ++p) {
            int i = ATi[p];
            process_edge(j, i, first, maxfirst, delta, prevleaf, ancestor);
        }

        if (parent[j] != -1)
            ancestor[j] = parent[j]; // union-find parent update
    }

    // Sum deltas up the tree to get final counts (children contribute to parent)
    colcount = delta;
    for (int j = 0; j < n; ++j) {
        int pj = parent[j];
        if (pj != -1)
            colcount[pj] += colcount[j];
    }

    return colcount; // colcount[j] = nnz in column j of L (incl. diagonal)
}

/**
 * @brief Find the symbolic cholesky factorization
 */
template <typename T>
SChol schol(const csc_matrix<T, sym::upper>& A) {
    const int n = static_cast<int>(A.size());
    SChol S;

    // etree on A (upper triangle)
    S.parent = etree(A);

    // Reserve column pointers
    S.cp.resize(n + 1);
    S.cp[0] = 0;

    std::vector<int> s(n), w(n, -1);
    std::vector<T> x(n, T(0));

    std::vector<int> rowind_all; // will hold all row indices concatenated

    for (int j = 0; j < n; ++j) {
        const auto top = ereach(A, j, S.parent, s, w, x, n);

        // Start a new column
        int col_start = static_cast<int>(rowind_all.size());

        // Always include diagonal
        rowind_all.push_back(j);

        // Include reach set
        for (auto t = top; t < n; ++t) {
            int row = s[t];
            if (row > j) { // enforce lower part only
                rowind_all.push_back(row);
            }
        }

        // sort for consistency
        std::sort(rowind_all.begin() + col_start, rowind_all.end());

        // Update column pointer
        S.cp[j + 1] = static_cast<int>(rowind_all.size());
    }

    // Finalize row indices
    S.rowind = std::move(rowind_all);

    const auto nz = S.rowind.size();

    S.lnz = nz;

    // not used for the cholesky, but I have it here for completeness
    S.unz = nz;

    return S;
}

namespace internal {
    // Compute the nonzero pattern of column k of L.
    // On return, s[top..n-1] contains the row indices (in topological order).
    // Returns new top.
    // n = A.size()
    // parent = etree(A)
    // s, w, x should be preallocated length n.

    template <typename T, typename Updater>
    std::size_t ereach_impl(const csc_matrix<T>& A, std::size_t k, const std::vector<int>& parent, std::vector<int>& s, std::vector<int>& w, std::size_t top, Updater&& update) {

        const auto& Ap = A.p();
        const auto& Ai = A.i();

        for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
            int i = Ai[p];
            if (i > k)
                continue; // upper part only

            update(p, i);

            std::vector<int> path; // stack
            while (i != -1 && w[i] != k) {
                path.push_back(i);
                w[i] = static_cast<int>(k);
                i = parent[i];
            }

            while (!path.empty()) {
                s[--top] = path.back();
                path.pop_back();
            }
        }

        return top;
    }
}

/**
 * @brief Compute the nonzero pattern of column k of L (has numerical accumulation)
 * @param k column
 * @param parent elimination tree
 * @param top number of columns/rows
 */
template <typename T>
std::size_t ereach(const csc_matrix<T>& A, std::size_t k, const std::vector<int>& parent, std::vector<int>& s, std::vector<int>& w, std::vector<T>& x, std::size_t top) {
    const auto& Ax = A.x();
    return internal::ereach_impl(A, k, parent, s, w, top, [&](int p, int i) { x[i] = Ax[p]; });
}

/**
 * @brief Compute the nonzero pattern of column k of L (purely symbolic)
 * @param k column
 * @param parent elimination tree
 * @param top number of columns/rows
 */
template <typename T>
std::size_t ereach(const csc_matrix<T>& A, std::size_t k, const std::vector<int>& parent, std::vector<int>& s, std::vector<int>& w, std::size_t top) {
    return internal::ereach_impl(A, k, parent, s, w, top, [](int, int) { });
}

std::vector<std::vector<int>> compute_levels(const std::vector<int>& parent);

/**
 * @brief Calculate the cholesky
 * @param A Upper triangle of csc_matrix A
 * @param S parent, cp from your symbolic step
 * @param optional status
 */
template <typename T>
std::expected<csc_matrix<T, sym::lower>, std::string> chol(const csc_matrix<T, sym::upper>& A) {
    const std::size_t n = A.size();

    const auto parent = etree(A);

    /// postorder of the etree
    const auto post = post_order(parent);

    /// column counts for L (includes diagonal)
    const auto colcount = col_count(A, parent, post);

    /// column pointers
    std::vector<int> cp(n + 1);

    // column pointers are created by accumulating the column count
    int nz = 0;
    for (int j = 0; j < n; ++j) {
        cp[j] = nz;
        nz += colcount[j];
    }

    cp[n] = nz;

    assert(cp.size() == n + 1);
    assert(parent.size() == n);

    // Allocate result matrix `L` (lower triangular, stored in CSC form)
    csc_matrix<T, sym::lower> L(n, cp.back());
    L.p() = cp;

    auto& Li = L.i();
    auto& Lx = L.x();

    /// `c[i]` holds the next free slot for column i
    std::vector<std::atomic<int>> c(n);
    for (int j = 0; j < n; ++j) {
        c[j].store(cp[j], std::memory_order_relaxed);
    }

    const auto levels = compute_levels(parent);

    for (auto lvl : levels) {

#pragma omp parallel for schedule(dynamic)

        for (int idx = 0; idx < lvl.size(); ++idx) {

            /// `k` is the index of the column that we want to factor
            const auto k = lvl[idx];

            // Work arrays
            std::vector<int> s(n, -1);
            std::vector<int> w(n, -1);
            std::vector<T> x(n, T(0));

            L.p()[k] = c[k];
            x[k] = T(0);
            w[k] = k;

            /// `ereach` fills `s[top..n-1]` with nonzero pattern and accumulates `x[]`
            const auto top = ereach(A, k, parent, s, w, x, n);

            /// d is the diagonal accumulator.
            /// At the beginning its assigned $A_kk$.
            /// But at the end of the loop it looks like
            /// $d = A_{kk{} - \sum L_{ik}^2$
            T d = x[k];
            x[k] = T(0);

            for (auto t = top; t < n; t++) {

                // row `i`
                int i = s[t];

                const T Lii = Lx[L.p()[i]]; // first entry in col i is diagonal
                assert(Lii != T(0));

                T lki = x[i] / Lii;
                x[i] = T(0);

                // Accumulate updates
                // loops over every nonzero column
                // `L.p()[i]` points to the start of the column `i` (where the diag is)
                // `c[i]` is the current end
                // so the nonzero off diagonal entries are between `{L.p()[i], c[i] - 1}`
                for (int p = L.p()[i] + 1; p < c[i]; ++p) {
                    // Li[p] is the row index stored at position p
                    // Lx[p] is the numeric value of that entry
                    x[Li[p]] -= Lx[p] * lki;
                }

                d -= lki * lki;

                // Append entry L(k,i) into column i
                int q = c[i].fetch_add(1, std::memory_order_acq_rel);
                Li[q] = k;
                Lx[q] = lki;
            }

            if (d <= T(0)) {
                return std::unexpected("A is not positive definite.");
            }

            // Place diagonal entry L(k,k)
            int q = c[k].fetch_add(1, std::memory_order_acq_rel);
            Li[q] = k;
            Lx[q] = std::sqrt(d);
        }
    }

    // finalize last column pointer
    L.p()[n] = cp[n];

    return L;
}

template <typename T>
SChol s_chol(const csc_matrix<T, sym::upper>& A) {
    const auto n = A.size();

    SChol S;

    S.parent = etree(A);

    /// postorder of the etree
    const auto post = post_order(S.parent);

    /// column counts for `L` (includes diagonal)
    const auto colcount = col_count(A, S.parent, post);

    /// column pointers
    std::vector<int> cp(n + 1);

    // column pointers are created by accumulating the column count
    int nz = 0;
    for (int j = 0; j < n; ++j) {
        cp[j] = nz;
        nz += colcount[j];
    }

    cp[n] = nz;

    assert(S.parent.size() == n);

    S.cp = cp;

    auto& Si = S.rowind;
    Si.assign(nz, 0);

    /// `c[i]` holds the next free slot for column i
    std::vector<std::atomic<int>> c(n);
    for (int j = 0; j < n; ++j) {
        c[j].store(cp[j], std::memory_order_relaxed);
    }

    const auto levels = compute_levels(S.parent);

    for (auto lvl : levels) {

#pragma omp parallel for schedule(dynamic)

        for (int idx = 0; idx < lvl.size(); ++idx) {

            /// `k` is the index of the column that we want to factor
            const auto j = lvl[idx];

            // Work arrays
            std::vector<int> s(n, -1);
            std::vector<int> w(n, -1);
            std::vector<T> x(n, T(0));

            S.cp[j] = c[j];
            x[j] = T(0);
            w[j] = j;

            /// `ereach` fills `s[top..n-1]` with nonzero pattern and accumulates `x[]`
            const auto top = ereach(A, j, S.parent, s, w, x, n);

            for (auto t = top; t < n; ++t) {
                int i = s[t];

                // Append entry L(k,i) into column i
                int q = c[i].fetch_add(1, std::memory_order_acq_rel);
                Si[q] = j;
            }

            // Place diagonal entry L(k,k)
            int q = c[j].fetch_add(1, std::memory_order_acq_rel);
            Si[q] = j;
        }
    }

    // finalize last column pointer
    S.cp[n] = cp[n];

    return S;
}

/**
 * @brief Group columns into supernodes.
 * @param S The symbolic cholesky
 * @param supernodes gets filled with the start indices of each supernode.
 * @return std::vector<int> size `n`; where super[j] is the supernode id of column j
 */
std::vector<int> compute_supernodes(const SChol& S, std::vector<std::size_t>& supernodes);

/**
 * @brief Generate a random symmetric sparse matrix in CSC format (upper triangle stored).
 * @param n matrix size (square)
 * @param density percentage of nonzeros to zeros
 */
template <typename T>
csc_matrix<T, sym::upper> random_sparse(int n, double density = 0.25, bool positive_definite = true) {
    // I used this stack overflow answer as a base
    // https://stackoverflow.com/a/30742847

    pcg32 gen(21);

    // Choose a random mean between 1 and 6
    std::uniform_real_distribution<double> dist(-n, n);

    std::vector<int> ti; // row indices
    std::vector<int> tj; // col indices
    std::vector<T> tx; // values

    for (int i = 0; i < n; ++i) {
        for (int j = i; j < n; ++j) { // only upper triangle
            double v_ij = dist(gen);

            // this forces us to always keep the diagonal
            if (abs(v_ij) < (density * n) || i == j) {
                T val = static_cast<T>(dist(gen));

                if (i == j && positive_definite) {
                    val += static_cast<T>(n); // boost diagonal
                }

                ti.push_back(i);
                tj.push_back(j);
                tx.push_back(val);
            }
        }
    }

    return triplet_to_csc_matrix(ti, tj, tx, n);
}

/**
 * @brief Compute the rows in a supernode
 */
template <typename T>
std::vector<int> supernode_rows(const csc_matrix<T, sym::upper>& A, const std::vector<int>& parent, std::size_t start, std::size_t end) {

    const auto n = A.size();
    std::vector<int> s(n), w(n, -1);
    std::vector<T> x(n, T(0));

    std::vector<int> rows;
    for (auto k = start; k < end; ++k) {
        const auto top = ereach(A, k, parent, s, w, n);
        for (auto t = top; t < n; ++t) {
            rows.push_back(s[t]);
        }
    }

    // we take the union of all rows in the supernodes
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

    return rows;
}

template <typename T>
class panel {
private:
    /// column-major dense block
    std::vector<T> data;

    /// the range of columns we want to pull from
    /// since the columns are contiguous
    std::pair<std::size_t, std::size_t> column_range;

    /// global row indices (mapping back to CSC rows)
    std::vector<int> rows_;

    /// #rows in dense panel
    const size_t m_ = 0;

    /// #cols in dense panel
    const size_t n_ = 0;

public:
    panel(std::vector<int> rows, std::size_t start, std::size_t end) :
        m_(rows.size()), n_(end - start),
        data(rows.size() * (end - start), T(0)),
        rows_(rows), column_range({ start, end }) { }

    std::size_t nrows() const { return m_; }
    std::size_t ncols() const { return n_; }

    /// access element (col-major layout)
    T& operator()(std::size_t r, std::size_t c) { return data[c * m_ + r]; }
    const T& operator()(std::size_t r, std::size_t c) const { return data[c * m_ + r]; }

    /// data getter helpers
    T* data_ptr() { return data.data(); }
    const T* data_ptr() const { return data.data(); }

    std::vector<int>& get_rows() { return rows_; }
    const std::vector<int>& get_rows() const { return rows_; }

    const std::pair<std::size_t, std::size_t> get_column_range() const { return column_range; }
};

/**
 * @brief Extract dense panel from a collection of supernodes
 * @param L
 * @param start
 * @param end
 * @return vector<T>
 */
template <typename T, sym S>
panel<T> extract_panel(const csc_matrix<T, S>& L, std::size_t start, std::size_t end, const std::vector<int>& row_index) {
    // filter out eliminated rows (those < start)
    std::vector<int> filtered_rows;
    filtered_rows.reserve(row_index.size());
    for (int r : row_index) {
        if (r >= start) {
            filtered_rows.push_back(r);
        }
    }

    panel<T> P(filtered_rows, start, end);

    const auto& Lp = L.p();
    const auto& Li = L.i();
    const auto& Lx = L.x();

    const int m = static_cast<int>(L.rows());
    const int nr = static_cast<int>(filtered_rows.size());

    // Global row to local row lookup
    std::vector<int> row2local(m, -1);
    for (int r = 0; r < nr; ++r) {
        row2local[filtered_rows[r]] = r;
    }

    // Scatter each column `j` into dense panel
    for (auto j = start; j < end; ++j) {
        auto local_col = j - start;

        for (int p = Lp[j]; p < Lp[j + 1]; ++p) {
            int row = Li[p];
            std::size_t local_row = row2local[row];

            if (local_row != -1) {
                P(local_row, local_col) = static_cast<int>(Lx[p]);
            }
        }
    }

    return P;
}

struct UpdateBlock {
    /// rows affected by the update (global indices), these are the rows "below" the supernode
    std::vector<int> rows;

    /// dense lower-symmetric matrix in column-major (mb x mb), representing -(L_{rect} * L_{rect}^T)
    std::vector<double> C;

    int ld = 0; // leading dimension for C (== mb)
};

/**
 * @brief Scatter one dense panel column back into global CSC L
 * @param j column index in L
 * @param rows row indices of the panel in L
 * @param Pcol pointer to dense panel column (length m, ld=m)
 */
template <typename T>
inline void scatter_panel_column_into_L(csc_matrix<T, sym::lower>& L, std::size_t j, const std::vector<int>& rows, const T* Pcol) {
    // For each row in the panel, directly update L(row, j)
    for (std::size_t r = 0; r < rows.size(); ++r) {
        if (Pcol[r] != 0) {
            const int row = rows[r];
            L[row, j] = Pcol[r];
        }
    }
}

/**
 * @brief Apply the trailing update block C into the global sparse matrix A.
 * @param A csc_matrix; upper triangular only
 */
template <typename T>
void apply_update(csc_matrix<T, sym::upper>& A, const UpdateBlock& upd) {
    const int mb = upd.ld;
    if (mb <= 0)
        return;

    const auto& rows = upd.rows;

    for (int jj = 0; jj < mb; ++jj) {
        const int col = rows[jj];
        for (int ii = jj; ii < mb; ++ii) {
            const int row = rows[ii];

            const int idx = jj * mb + ii;

            if (upd.C[idx] != 0) {
                // add contribution into A(row, col)
                A[row, col] += upd.C[jj * mb + ii]; // C is column-major
            }
        }
    }
}

/**
 * @brief Compute the assembly tree
 */
std::vector<int> atree(const SChol& S, const std::vector<int>& sn_id, const std::vector<std::size_t>& supernodes);

/**
 * @brief Factorize a supernode
 * @param start supernode starting column
 * @param end supernode ending column `[start, end)`
 * @param P assembled dense panel (col-major)
 * @param S symbolic output (cp)
 * @param L global L to update in-place
 */
template <typename T>
std::expected<UpdateBlock, std::string> factorize_sn(std::size_t start, std::size_t end, panel<T>& P, const SChol& S, csc_matrix<T, sym::lower>& L) {

    /// super node width (in columns)
    const auto w = end - start;

    /// #rows in panel
    const auto m = P.nrows();

    /// #below diagonal rows
    const auto mb = m - w;

    const auto& rows = P.get_rows();

    if (w <= 0)
        return std::unexpected("Empty supernode.");

    // L_{diag}
    // We find `L_{diag}` by performing the Cholesky on `A_{diag}`
    {
        char uplo = 'L';

        /// order of the matrix (supernode width)
        auto nblk = static_cast<int>(w);

        /// leading dimension of the panel (rows)
        /// lda >= nblk
        auto lda = static_cast<int>(m);

        int info = 0;

        // performs cholesky factorization on L_{diag}
        dpotrf_(&uplo, &nblk, P.data_ptr(), &lda, &info);

        if (info != 0) {
            if (info < 0) {
                return std::unexpected(
                    "dpotrf_ failed: the " + std::to_string(-info) + "-th argument had an illegal value.");
            } else {
                // info > 0: breakdown at diagonal element `info`
                std::stringstream ss;
                ss << "dpotrf_ failed in supernode [" << start << "," << end
                   << "): leading minor of order " << info
                   << " is not positive definite.";
                // Optionally: dump the diagonal block for debugging
                ss << " Diagonal entries: ";
                for (int d = 0; d < nblk; ++d) {
                    ss << P(d, d) << " ";
                }
                return std::unexpected(ss.str());
            }
        }
    }

    // L_{rect}
    /// We find `L_{rect}` using a triangular solve on the equation
    /// `L_{rect} = A_{rect} L_{diag}^{-T}`
    if (mb > 0) {
        const double* L_diag = P.data_ptr(); // (w x w), ld = m
        double* A_rect = P.data_ptr() + w; // (mb x w), ld = m  (row offset = w)

        cblas_dtrsm(CblasColMajor,
                    CblasRight, // B := B * inv(op(A))
                    CblasLower, // A is lower (L11)
                    CblasTrans, // op(A) = L11^T
                    CblasNonUnit,
                    static_cast<int>(/*m=*/mb), // rows of B
                    static_cast<int>(/*n=*/w), // cols of B
                    /*alpha=*/1.0,
                    L_diag,
                    static_cast<int>(/*lda=*/m),
                    A_rect,
                    static_cast<int>(/*ldb=*/m));
    }

    // Scatter both diagonal & rectangular pieces back into L's CSC columns
    for (auto j = start; j < end; ++j) {
        const auto cj = j - start;

        /// Address of column `j` in panel
        const auto* Pcol = P.data_ptr() + cj * m;

        scatter_panel_column_into_L(L, j, rows, Pcol);
    }

    // Build the trailing update: A_{off} := A_{off} - L_{rect} * L_{rect}^T (lower, dense)
    UpdateBlock upd;
    upd.ld = std::max(0, static_cast<int>(mb));

    if (mb > 0) {
        upd.rows.assign(rows.begin() + w, rows.end()); // rows below the supernode
        upd.C.assign(static_cast<size_t>(mb) * mb, 0.0);

        const double* L_rect = P.data_ptr() + w; // (mb x w), ld = m

        // C := -1 * L_{rect} * L_{rect}^T + 0*C
        cblas_dsyrk(CblasColMajor, CblasLower,
                    CblasNoTrans, // L21 is mb x w
                    static_cast<int>(/*N=*/mb),
                    static_cast<int>(/*K=*/w),
                    /*alpha=*/-1.0,
                    L_rect,
                    static_cast<int>(/*lda=*/m),
                    /*beta=*/0.0,
                    upd.C.data(),
                    static_cast<int>(/*ldc=*/mb));
    }

    return upd;
}

template <typename T>
std::expected<csc_matrix<T, sym::lower>, std::string> chol_sn(csc_matrix<T, sym::upper>& A) {
    // Symbolic analysis
    const auto S = schol(A);

    // Allocate L with symbolic size
    csc_matrix<T, sym::lower> L(S);

    // Supernode partition
    std::vector<std::size_t> supernodes;
    const auto sn_id = compute_supernodes(S, supernodes);

    const auto at = atree(S, sn_id, supernodes);
    const auto levels = compute_levels(at);

    for (auto& lvl : levels) {
#pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < lvl.size(); ++idx) {
            const auto sn = lvl[idx];

            const auto start = supernodes[sn];
            const auto end = supernodes[sn + 1];

            // Compute rows spanned by this supernode
            const auto rows = supernode_rows(A, S.parent, start, end);

            panel<T> P = extract_panel(A.transpose(), start, end, rows);

            // Factorize this supernode
            const auto result = factorize_sn(start, end, P, S, L);
            if (!result)
                return std::unexpected(result.error());

            apply_update(A, result.value());
        }
    }

    return L;
}

/**
 * @brief Convert a CSC matrix (lower triangular) into an SChol structure.
 */
template <typename T>
SChol to_schol(const csc_matrix<T, sym::lower>& L) {
    SChol S;

    S.cp = L.p();
    S.rowind = L.i();
    S.lnz = L.i().size();
    S.unz = S.lnz;

    S.parent.assign(L.size(), -1);

    return S;
}
