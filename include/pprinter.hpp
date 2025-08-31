// pretty printer

#pragma once

#include <iomanip>

#include "chol.hpp"

template <typename T, sym S>
std::ostream& operator<<(std::ostream& os, const csc_matrix<T, S>& A) {
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(A.cols());

    int width = 8;
    int precision = 2;

    os << std::fixed << std::setprecision(precision);

    // Print column headers
    os << std::setw(width) << " ";
    for (int j = 0; j < n; ++j) {
        os << std::setw(width) << j;
    }
    os << "\n";

    for (int i = 0; i < m; ++i) {
        os << std::setw(width) << i; // row label
        for (int j = 0; j < n; ++j) {
            os << std::setw(width) << A[i, j];
        }
        os << "\n";
    }

    return os;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const panel<T>& P) {
    int m = P.nrows();
    int n = P.ncols();
    const auto& rows = P.get_rows();

    int width = 10;
    int precision = 4;

    os << std::fixed << std::setprecision(precision);

    os << std::setw(width) << " ";
    for (int j = 0; j < n; ++j) {
        int global_col = P.get_column_range().first + j; // use stored start..end
        os << std::setw(width) << global_col;
    }
    os << "\n";

    for (int i = 0; i < m; ++i) {
        os << std::setw(width) << rows[i];
        for (int j = 0; j < n; ++j) {
            os << std::setw(width) << P(i, j);
        }
        os << "\n";
    }

    return os;
}

std::string to_string(const elimination_tree& parent);

std::ostream& operator<<(std::ostream& os, const SChol& S);