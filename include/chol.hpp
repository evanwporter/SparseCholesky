#pragma once

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <expected>
#include <random>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pcg_random.hpp"

enum class sym {
    none,
    upper,
    lower
};

using elimination_tree = std::vector<int>;

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

public:
    csc_matrix(int m, int n, int nzmax) :
        n_(n), m_(m), nzmax_(nzmax),
        p_(n + 1, 0), i_(nzmax), x_(nzmax, T(0)) {
        assert(m > 0 && n > 0 && nzmax > 0);

        if constexpr (S != sym::none) {

            // enforce square for symmetric
            assert(m == n);
        }
    }

    // Constructor only available if Symmetric
    csc_matrix(int n, int nzmax)
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

    // Access (i, j) with symmetry awareness
    T operator[](int i, int j) const {
        assert(i >= 0 && i < m_ && j >= 0 && j < n_);

        if constexpr (S == sym::upper) {
            if (j < i)
                std::swap(i, j);
        } else if constexpr (S == sym::lower) {
            if (i < j)
                std::swap(i, j);
        }

        for (int idx = p_[j]; idx < p_[j + 1]; ++idx) {
            if (i_[idx] == i)
                return x_[idx];
        }

        // If we can't find it in the column pointers
        return T(0);
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

/// Convert triplet format to symmetric positive definite (SPD) matrix
template <typename T>
csc_matrix<T, sym::upper> triplet_to_csc_matrix(const std::vector<int>& ti, const std::vector<int>& tj, const std::vector<T>& tx, int n) {
    /**
     * @brief Convert triplet format to symmetric positive definite (SPD) matrix
     * @param ti row indeices
     * @param tj column indices
     * @param tx values
     */

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

/// Compute the elimination tree of an SPD matrix (upper triangle stored).
/// Returns a vector<int> of size n, where parent[j] is the parent of column j.
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

/// Iterative DFS used by postorder
/// Taken from SuiteSparse (Dr. Tim Davis)
static int tdfs(int root, int k, std::vector<int>& head, const std::vector<int>& next, std::vector<int>& post, std::vector<int>& stack) {
    /**
     * @param next sibling links
     * @param post output posterorder
     * @param stack temporary stack (size >= n)
     */

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

// Postorder of a forest given parent[] (parent[j] == -1 for roots).
// Matches cs_post sibling ordering by linking children while scanning j=n-1..0.
inline std::vector<int> post_order(const elimination_tree& parent) {
    const int n = static_cast<int>(parent.size());
    assert(n >= 0);

    std::vector<int> post(n, -1);
    std::vector<int> head(n, -1); // head[v] = youngest child of v (singly-linked list)
    std::vector<int> next(n, -1); // next[s] = older sibling of s (in the list of parent[s])
    std::vector<int> stack(n, -1); // DFS stack

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

// Transpose (pattern only) of your CSC upper-tri matrix A
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

// Column counts for Cholesky (ata = 0 path of cs_counts)
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

struct SChol {
    /// elimination tree (size n)
    std::vector<int> parent;

    /// column pointers for L (size n+1)
    std::vector<int> cp;

    /// number nonzeros in lower triangle
    int lnz = 0; // nnz(L)

    /// number nonzeros in upper triangle
    int unz = 0;
};

template <typename T>
SChol schol(const csc_matrix<T, sym::upper>& A) {
    const int n = static_cast<int>(A.size());
    SChol S;

    // etree on A (upper triangle)
    S.parent = etree(A); // your function above

    // postorder of the etree
    auto post = post_order(S.parent);

    // column counts for L (includes diagonal)
    auto colcount = col_count(A, S.parent, post);

    // column pointers
    // done by accumulating the column count
    S.cp.resize(n + 1);
    int nz = 0;
    for (int j = 0; j < n; ++j) {
        S.cp[j] = nz;
        nz += colcount[j];
    }
    S.cp[n] = nz;

    S.lnz = nz;

    // not used for the cholesky, but I have it here for completeness
    S.unz = nz;
    return S;
}

// Compute the nonzero pattern of column k of L.
// On return, s[top..n-1] contains the row indices (in topological order).
// Returns new top.
// n = A.size()
// parent = etree(A)
// s, w, x should be preallocated length n.

template <typename T>
int ereach(const csc_matrix<T>& A, int k, const std::vector<int>& parent, std::vector<int>& s, std::vector<int>& w, std::vector<T>& x, int top) {
    /**
     * @brief Compute the nonzero pattern of column k of L.
     * @param k column
     * @param parent elimination tree
     */
    const auto& Ap = A.p();
    const auto& Ai = A.i();
    const auto& Ax = A.x();

    for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
        int i = Ai[p];
        if (i > k)
            continue; // upper part only
        x[i] = Ax[p];

        std::vector<int> path; // local stack
        while (i != -1 && w[i] != k) {
            path.push_back(i);
            w[i] = k;
            i = parent[i];
        }
        while (!path.empty()) {
            s[--top] = path.back();
            path.pop_back();
        }
    }
    return top;
}

std::vector<std::vector<int>> compute_levels(const std::vector<int>& parent);

/**
 * @brief Calculate the cholesky
 * @param A Upper triangle of csc_matrix A
 * @param S parent, cp from your symbolic step
 * @param optional status
 */
template <typename T>
std::expected<csc_matrix<T, sym::lower>, std::string> chol(const csc_matrix<T, sym::upper>& A, const SChol& S) {
    const int n = static_cast<int>(A.size());
    assert(static_cast<int>(S.cp.size()) == n + 1);
    assert(static_cast<int>(S.parent.size()) == n);

    // Allocate result matrix `L` (lower triangular, stored in CSC form)
    csc_matrix<T, sym::lower> L(n, S.cp.back());
    L.p() = S.cp;
    auto& Li = L.i();
    auto& Lx = L.x();

    /// `c[i]` holds the next free slot for column i
    std::vector<std::atomic<int>> c(n);
    for (int j = 0; j < n; ++j) {
        c[j].store(S.cp[j], std::memory_order_relaxed);
    }

    const auto& parent = S.parent;

    auto levels = compute_levels(S.parent);

    for (auto lvl : levels) {
#pragma omp parallel for schedule(dynamic) // or guided

        for (int idx = 0; idx < lvl.size(); ++idx) {

            /// `k` is the column that we want to factor
            const int k = lvl[idx];

            // Work arrays
            std::vector<int> s(n, -1);
            std::vector<int> w(n, -1);
            std::vector<T> x(n, T(0));

            L.p()[k] = c[k]; // keep consistency with CSparse
            x[k] = T(0);
            w[k] = k;

            /// `ereach` fills `s[top..n-1]` with nonzero pattern and accumulates `x[]`
            int top = ereach(A, k, parent, s, w, x, n);

            /// d is the diagonal accumulator.
            /// At the beginning its assigned $A_kk$.
            /// But at the end of the loop it looks like
            /// $d = A_{kk{} - \sum L_{ik}^2$
            T d = x[k];
            x[k] = T(0);

            for (; top < n; ++top) {
                int i = s[top];

                const T Lii = Lx[L.p()[i]]; // first entry in col i is diagonal
                assert(Lii != T(0));

                T lki = x[i] / Lii;
                x[i] = T(0);

                // Accumulate updates
                // loops over every nonzero column
                // L.p()[i] points to the start of the column i (where the diag is)
                // c[i] is the current end
                // so the nonzero off diagonal entries are between {L.p()[i], c[i] - 1}
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
    L.p()[n] = S.cp[n];

    return L;
}

/// Group columns into supernodes.
/// Returns a vector<int> of size n, where super[j] is the supernode id of column j.
/// Also fills `supernodes` with the start indices of each supernode.
std::vector<int> compute_supernodes(const SChol& S, std::vector<int>& supernodes);

/**
 * Generate a random symmetric sparse matrix in CSC format (upper triangle stored).
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