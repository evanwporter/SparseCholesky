// pretty printer

#pragma once

#include <iomanip>
#include <sstream>

#include "chol.hpp"

template <typename T, sym S>
std::string to_string(const csc_matrix<T, S>& A, int width = 8, int precision = 2) {
    std::stringstream stream;
    stream << std::fixed << std::setprecision(precision);

    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            stream << std::setw(width) << A[i, j] << " ";
        }
        stream << "\n";
    }

    return stream.str();
}

template <typename T>
std::string to_string_sparsity_pattern(const csc_matrix<T>& A) {
    std::stringstream stream;

    const int n = A.size();

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[i, j] != 0)
                stream << "*  ";
            else
                stream << "   ";
        }
        stream << "\n";
    }

    return stream.str();
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
