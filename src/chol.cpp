#include <algorithm>
#include <vector>

#include <chol.hpp>

// parent[j] = parent of j, or -1 if root
std::vector<std::vector<int>> compute_levels(const std::vector<int>& parent) {
    const auto n = parent.size();
    std::vector<int> depth(n, -1);

    for (int j = 0; j < n; ++j) {
        if (depth[j] != -1)
            continue;

        // climb to a known depth or root
        int v = j;
        std::vector<int> path;
        while (v != -1 && depth[v] == -1) {
            path.push_back(v);
            v = parent[v];
        }
        int base = (v == -1 ? 0 : depth[v] + 1);

        // path compression: assign depths on the way back
        for (int k = static_cast<int>(path.size()) - 1; k >= 0; --k) {
            depth[path[k]] = base++;
        }
    }

    int maxd = 0;
    for (int d : depth)
        maxd = std::max(maxd, d);
    std::vector<std::vector<int>> levels(maxd + 1);
    for (int j = 0; j < n; ++j)
        levels[depth[j]].push_back(j);

    std::reverse(levels.begin(), levels.end());

    return levels;
}

std::vector<int> compute_supernodes(const SChol& S, std::vector<std::size_t>& supernodes) {
    const auto n = S.parent.size();
    const auto& parent = S.parent;
    const auto& cp = S.cp; // column pointers

    /// sn_id maps each column to its supernode id
    std::vector<int> sn_id(n, -1);

    // `supernodes` is like the column pointers
    // it points to the column where a supernode starts
    // ie: supernodes = [0, 2, 5, ...]
    //     means there are supernodes spanning columns
    //     [0..1], [2..4], etc...

    supernodes.reserve(n);
    supernodes.clear();
    supernodes.push_back(0); // first supernode starts at col 0

    int sid = 0; // current supernode id
    sn_id[0] = sid;

    for (int j = 1; j < n; ++j) {
        bool same_pattern = false;

        // supernode test:
        // (1) parent[j] == j + 1  (j+1 is parent of j in etree)
        // (2) nnz(L(:,j)) - nnz(L(:,j-1)) == 1
        //       (they have the same # of nonzeros below the diagonal
        //        minus 1 for the diagonal)
        // once these two conditions are met then we know they have the same
        // sparsity structure below the diagonal. Since the principle of column
        // replication ensures that the sparsity pattern of column j is replicated
        // into column j + 1 if j + 1 is the parent of j
        if (parent[j - 1] == j) {
            /// non zeros in the jth column
            int lenj = cp[j + 1] - cp[j];

            /// non zeros in the j - 1 column
            int lenjm1 = cp[j] - cp[j - 1];

            // -1 is to account for the diagonal
            if (lenj == lenjm1 - 1)
                same_pattern = true;
        }

        if (same_pattern) {
            // merge j into current supernode
            sn_id[j] = sid;
        } else {
            // start new supernode
            supernodes.push_back(j);
            sid++;
            sn_id[j] = sid;
        }
    }

    supernodes.push_back(n);
    return sn_id;
}

std::vector<int> atree(const SChol& S, const std::vector<int>& sn_id, const std::vector<std::size_t>& supernodes) {

    /// #columns
    const auto n = S.parent.size();

    /// #supernodes
    const auto ns = supernodes.size() - 1;

    assert((int)sn_id.size() == n);
    assert(supernodes.front() == 0 && supernodes.back() == n);

    std::vector<int> super_parent(ns, -1);

    // Build parent relation for supernodes
    for (int s = 0; s < ns; ++s) {
        const auto start = supernodes[s];
        const auto end = supernodes[s + 1]; // exclusive

        for (auto j = start; j < end; ++j) {
            for (auto p = S.cp[j]; p < S.cp[j + 1]; ++p) {
                int row = S.rowind[p];
                if (row >= end) {
                    int t = sn_id[row];
                    if (t != s) {
                        if (super_parent[s] == -1 || t < super_parent[s]) {
                            super_parent[s] = t;
                        }
                    }
                }
            }
        }
    }

    return super_parent;
}