#include <iomanip>
#include <iostream>

#include "chol.hpp"
#include "pprinter.hpp"

std::ostream& operator<<(std::ostream& os, const SChol& S) {
    int n = static_cast<int>(S.parent.size());
    os << "Symbolic structure (n=" << n << ", nnz=" << S.cp.back() << ")\n";

    // Print column headers
    os << std::setw(4) << " ";
    for (int j = 0; j < n; ++j) {
        os << std::setw(2) << j;
    }
    os << "\n";

    // For each row i
    for (int i = 0; i < n; ++i) {
        os << std::setw(4) << i;
        for (int j = 0; j < n; ++j) {
            // scan column j's row list
            bool found = false;
            for (int p = S.cp[j]; p < S.cp[j + 1]; ++p) {
                if (S.rowind[p] == i) {
                    found = true;
                    break;
                }
            }
            os << std::setw(2) << (found ? '*' : '.');
        }
        os << "\n";
    }
    return os;
}

std::string to_string(const elimination_tree& parent) {
    std::stringstream stream;

    stream << "[";

    for (int v : parent) {
        stream << v << ", ";
    }

    stream << "]\n";

    return stream.str();
}