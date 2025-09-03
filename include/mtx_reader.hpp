#include <cassert>
#include <fstream>
#include <string>
#include <vector>

#include "chol.hpp"

/**
 * @brief Read matrix market file to `csc_matrix`
 * @note https://stackoverflow.com/a/57076216
 *
 * @tparam T
 * @param filename
 * @return csc_matrix<T, sym::upper>
 */
template <typename T>
csc_matrix<T, sym::upper> load_matrix_market_to_csc(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open file " + filename);
    }

    int num_row, num_col, num_lines;

    // Ignore comments headers
    while (file.peek() == '%')
        file.ignore(2048, '\n');

    // Read number of rows and columns
    file >> num_row >> num_col >> num_lines;

    assert(num_row == num_col && "Matrix must be square for Cholesky");

    std::vector<int> ti;
    std::vector<int> tj;
    std::vector<T> tx;

    ti.reserve(num_lines);
    tj.reserve(num_lines);
    tx.reserve(num_lines);

    // fill the matrix with data
    for (int l = 0; l < num_lines; ++l) {
        int row, col;
        T data;
        file >> row >> col >> data;

        row--;
        col--;

        if (col < row)
            std::swap(row, col);

        ti.push_back(row);
        tj.push_back(col);
        tx.push_back(data);
    }

    file.close();

    return triplet_to_csc_matrix(ti, tj, tx, num_row);
}
