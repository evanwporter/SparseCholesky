// pretty printer

#pragma once

#include <iomanip>

#include "chol.hpp"

template <typename T, sym S>
std::ostream& operator<<(std::ostream& os, const csc_matrix<T, S>& A) {
    const auto m = A.rows();
    const auto n = A.cols();

    constexpr int width = 8;
    constexpr int precision = 2;

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
    auto m = P.nrows();
    auto n = P.ncols();
    const auto& rows = P.get_rows();

    int width = 10;
    int precision = 4;

    os << std::fixed << std::setprecision(precision);

    os << std::setw(width) << " ";
    for (int j = 0; j < n; ++j) {
        const auto global_col = P.get_column_range().first + j;
        os << std::setw(width) << global_col;
    }
    os << "\n";

    for (int i = 0; i < m; ++i) {
        os << std::setw(width) << rows[i];
        for (int j = 0; j < n; ++j) {
            os << std::setw(width) << P[i, j];
        }
        os << "\n";
    }

    return os;
}

std::string to_string(const elimination_tree& parent);

std::string to_string(const std::vector<std::size_t>& parent);

std::ostream& operator<<(std::ostream& os, const SChol& S);